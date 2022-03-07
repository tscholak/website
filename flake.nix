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
    slick = {
      url = "github:ChrisPenner/slick?ref=51d4849e8fe3dad93ef8c10707975068f797fd28";
      flake = false;
    };
    llvm-hs = {
      url = "github:llvm-hs/llvm-hs?ref=llvm-12";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, haskell-nix, utils, ... }:
    utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          haskell-nix.overlay
          (final: prev: {
            # llvm-config = prev.llvmPackages_12.llvm;
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
                shell.additional = ps: [
                  ps.slick
                  ps.llvm-hs-pure
                  ps.llvm-hs
                ];
                cabalProjectLocal = ''
                  packages:
                    ${inputs.slick}
                    ${inputs.llvm-hs}/llvm-hs-pure
                    ${inputs.llvm-hs}/llvm-hs
                '';
                modules = [
                  {
                    packages.slick.src = inputs.slick.outPath;
                    packages.llvm-hs-pure.src = "${inputs.llvm-hs.outPath}/llvm-hs-pure";
                    packages.llvm-hs.src = "${inputs.llvm-hs.outPath}/llvm-hs";
                    packages.llvm-hs.components."library".build-tools = [
                      # final.hsPkgs.buildPackages.hsc2hs.components.exes.hsc2hs
                      final.buildPackages.llvm_12
                    ];
                  }
                ];
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
        check = pkgs.runCommand "combined-test"
          {
            checksss = builtins.attrValues flake.checks.${system};
          } ''
            echo $checksss
            touch $out
          '';
      });
}
