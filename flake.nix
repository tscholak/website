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
    utils.lib.eachSystem [ "x86_64-darwin" "x86_64-linux" ] (system:
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
      in rec {
        packages = {
          inherit website;
        };

        apps.website = utils.lib.mkApp {
          drv = packages.website;
          exePath = "/bin/website";
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

        defaultPackage = packages.website;

        defaultApp = apps.website;
      });
}
