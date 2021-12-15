{
  description = "website";

  nixConfig = {
    substituters = [
      https://hydra.iohk.io
    ];
    trusted-public-keys = [
      hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ=
    ];
    bash-prompt = "\\[\\033[1m\\][dev-website]\\[\\033\[m\\]\\040\\w$\\040";
  };

  inputs = {
    nixpkgs.follows = "haskell-nix/nixpkgs-unstable";
    haskell-nix = {
      url = "github:input-output-hk/haskell.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    utils.follows = "haskell-nix/flake-utils";
    iohkNix = {
      url = "github:input-output-hk/iohk-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, haskell-nix, utils, iohkNix, tokenizers, ... }:
    let
      inherit (nixpkgs) lib;
      inherit (lib) mapAttrs getAttrs attrNames;
      inherit (utils.lib) eachSystem;
      inherit (iohkNix.lib) prefixNamesWith collectExes;

      supportedSystems = ["x86_64-linux" "x86_64-darwin"];

      gitrev = self.rev or "dirty";

      overlays = [
        haskell-nix.overlay
        iohkNix.overlays.haskell-nix-extra

        (final: prev: {
          inherit gitrev;
          commonLib = lib
            // iohkNix.lib;
        })

        (final: prev: {
          website-project =
            let
              src = final.haskell-nix.haskellLib.cleanGit {
                name = "website";
                src = ./.;
              };
              compiler-nix-name = "ghc921";
              projectPackages = lib.attrNames (final.haskell-nix.haskellLib.selectProjectPackages
                (final.haskell-nix.cabalProject' {
                  inherit src compiler-nix-name;
                }).hsPkgs);
            in
              final.haskell-nix.cabalProject' {
                inherit src compiler-nix-name;

                modules = [
                  {
                    packages.website.enableExecutableProfiling = true;
                    enableLibraryProfiling = true;
                  }

                  (lib.optionalAttrs final.stdenv.hostPlatform.isMusl (let
                    fullyStaticOptions = {
                      enableShared = false;
                      enableStatic = true;
                      dontStrip = false;
                    };
                  in
                    {
                      packages = lib.genAttrs projectPackages (name: fullyStaticOptions);
                      doHaddock = false;
                    }
                  ))
                ];
              };
        })
      ];

    in eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs { inherit system overlays; };

        inherit (pkgs.commonLib) eachEnv environments;

        devShell =  pkgs.callPackage ./shell.nix {};

        flake = pkgs.website-project.flake {};

        staticFlake = pkgs.pkgsCross.musl64.website-project.flake {};

        exes = collectExes flake.packages;
        exeNames = attrNames exes;
        lazyCollectExe = p: getAttrs exeNames (collectExes p);

        packages = {
          inherit (pkgs) website;
        }
        // exes
        // (prefixNamesWith "static/"
              (mapAttrs pkgs.rewriteStatic (lazyCollectExe staticFlake.packages)));

      in lib.recursiveUpdate flake {
        inherit environments packages;

        defaultPackage = flake.packages."website:exe:website";

        defaultApp = utils.lib.mkApp { drv = flake.packages."website:exe:website"; };

        inherit devShell;
      }
    );
}