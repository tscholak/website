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
                  cabal = {};
                  haskell-language-server = {};
                };
                shell.buildInputs = with pkgs; [
                  nodePackages.serve
                  pandoc
                  nixpkgs-fmt
                ];
                shell.exactDeps = true;
              };
          })
        ];
        pkgs = import nixpkgs { inherit system overlays; inherit (haskell-nix) config; };
        flake = pkgs.websiteProject.flake {};
      in flake // rec {
        defaultPackage = flake.packages."website:exe:website";
        apps.website = utils.lib.mkApp {
          drv = defaultPackage;
          exePath = "/bin/website";
        };
        defaultApp = apps.website;
      });
}
