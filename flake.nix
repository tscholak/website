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
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    slick = {
      url = "github:ChrisPenner/slick?ref=51d4849e8fe3dad93ef8c10707975068f797fd28";
      flake = false;
    };
    llvm-hs = {
      url = "github:llvm-hs/llvm-hs?ref=llvm-12";
      flake = false;
    };
    dex-lang = {
      url = "github:google-research/dex-lang?ref=a80b2b0c53356817ea1252be546a9f560d4d9d9f";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, haskell-nix, utils, ... }:
    utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          haskell-nix.overlay
          (final: prev: {
            websiteProject =
              final.haskell-nix.project' {
                src = pkgs.haskell-nix.haskellLib.cleanGit {
                  name = "website";
                  src = ./.;
                };
                compiler-nix-name = "ghc8107";
                shell.tools = {
                  cabal = { };
                  haskell-language-server = { };
                };
                shell.buildInputs = with pkgs; [
                  nodePackages.serve
                  pandoc
                  nixpkgs-fmt
                ];
                shell.exactDeps = true;
                shell.additional = ps: [
                  ps.slick
                  ps.llvm-hs-pure
                  ps.llvm-hs
                  ps.dex
                ];
                cabalProjectLocal = ''
                  packages:
                    ${inputs.slick}
                    ${inputs.llvm-hs}/llvm-hs-pure
                    ${inputs.llvm-hs}/llvm-hs
                    ${inputs.dex-lang}
                '';
                modules = [
                  {
                    packages.slick.src = inputs.slick.outPath;
                    packages.llvm-hs-pure.src = "${inputs.llvm-hs.outPath}/llvm-hs-pure";
                    packages.llvm-hs.src = "${inputs.llvm-hs.outPath}/llvm-hs";
                    packages.llvm-hs.components."library".build-tools = [
                      final.buildPackages.llvm_12
                    ];
                    # packages.llvm-hs.configureFlags = [
                    #   "--with-gcc=${final.clang_12}/bin/cc"
                    #   "--with-ld=${final.clang_12}/bin/ld"
                    # ];
                    packages.dex.src =
                      final.runCommand "build-dexrt" { } ''
                        mkdir -p $out
                        cp -r ${inputs.dex-lang.outPath}/* $out
                        chmod u+w $out/src/lib
                        set -x
                        ${final.clang_12}/bin/clang++ \
                          -fPIC -std=c++11 -fno-exceptions -fno-rtti \
                          -c -emit-llvm \
                          -I ${final.libpng}/include \
                          $out/src/lib/dexrt.cpp \
                          -o $out/src/lib/dexrt.bc
                        set +x
                        chmod u-w $out/src/lib/dexrt.bc
                        chmod u-w $out/src/lib
                      '';
                    packages.dex.doHaddock = false;
                  }
                ];
              };
          })
        ];
        pkgs = import nixpkgs { inherit system overlays; inherit (haskell-nix) config; };
        flake = pkgs.websiteProject.flake { };
      in
      flake // rec {
        defaultPackage = flake.packages."website:exe:website";
        apps.website = utils.lib.mkApp {
          drv = defaultPackage;
          exePath = "/bin/website";
        };
        defaultApp = apps.website;
        check = pkgs.runCommand "combined-test"
          {
            checksss = builtins.attrValues flake.checks.${system};
          } ''
          echo $checksss
          touch $out
        '';
      });
}
