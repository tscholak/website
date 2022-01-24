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
  };

  outputs = { self, nixpkgs, haskell-nix, utils, ... }:
    utils.lib.eachSystem [ "x86_64-darwin" ] (system:
      let
        pkgs = haskell-nix.legacyPackages.${system};
        hsPkgs = pkgs.haskellPackages;

        haskellNix = pkgs.haskell-nix.cabalProject {
          src = pkgs.haskell-nix.haskellLib.cleanGit {
            name = "website";
            src = ./.;
          };
          compiler-nix-name = "ghc8107";
        };

        website = haskellNix.website.components.exes.website;

        build = pkgs.stdenv.mkDerivation {
          name = "tscholak-github-io-${self.shortRev or "dirty"}";

          src = ./.;

          dontBuild = true;

          installPhase = ''
            find .
            ${website}/bin/website --input-folder site --output-folder $out -f config.json
          '';
        };
      in {
        packages = {
          inherit website build;
        };

        devShell = haskellNix.shellFor {
          packages = p: [ p.website ];
          withHoogle = false;
          tools = {
            cabal = "latest";
            haskell-language-server = "latest";
          };
          nativeBuildInputs = [
            haskellNix.website.project.roots
            pkgs.nodePackages.serve
            pkgs.pandoc
          ];
          exactDeps = true;
        };

        defaultPackage = build;
      });
}
