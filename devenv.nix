{ pkgs
, lib
, config
, inputs
, ...
}:

{
  cachix.push = "tscholak";

  claude.code.enable = true;

  packages = [
    pkgs.git
    pkgs.nodejs
    pkgs.nodePackages.serve
    pkgs.watchexec
  ];

  languages.haskell = {
    enable = true;
    package = pkgs.haskell.compiler.ghc910;
    stack.enable = true;
    languageServer = lib.mkIf (config.devenv.isTesting) null;
  };
  languages.nix.enable = true;

  processes =
    { }
    // lib.optionalAttrs (!config.devenv.isTesting) {
      serve.exec = "npx serve build";
      build.exec = ''
        ${pkgs.watchexec}/bin/watchexec \
          --watch app \
          --watch doctests \
          --watch site \
          --watch config.json \
          --watch Setup.hs \
          --watch stack.yaml \
          --watch website.cabal \
          --restart \
          -- devenv-generate-build
      '';
    };

  scripts = {
    "devenv-generate-build" = {
      description = "Generate build artifacts for the website";
      exec = ''
        echo "Generating build artifacts...";
        stack run -- -f config.json
      '';
    };
    "devenv-test" = {
      description = "Run tests for the website";
      exec = ''
        echo "Running tests...";
        stack test --fast
      '';
    };
  };

  tasks = {
    "website:test" = {
      exec = "stack test";
      before = [ "devenv:enterTest" ];
    };
    "website:build".exec = "stack build";
    "website:cleanup" = {
      exec = "rm -rf build";
      after = [ "devenv:processes:serve" ];
    };
  };

  git-hooks.hooks = {
    nixpkgs-fmt.enable = true;
    ormolu = {
      enable = false;
      settings.defaultExtensions = [ "hs" ]; # add lhs files once ormolu supports them
    };
    cabal-gild.enable = true;
    markdownlint = {
      settings.configuration = {
        MD013 = {
          line_length = 120;
        };
        MD033 = false;
        MD034 = false;
      };
    };
  };
}
