{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.nodejs
    pkgs.nodePackages.serve
    pkgs.watchexec
  ];

  # https://devenv.sh/languages/
  languages.haskell = {
    enable = true;
    package = pkgs.haskell.compiler.ghc910;
    stack.enable = true;
  };
  languages.nix.enable = true;
  languages.python = {
    enable = true;
    version = "3.13";
    #   poetry.enable = true;
  };

  processes = {
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
  };

  tasks = {
    "website:test".exec = "stack test";
    "website:build".exec = "stack build";
    "website:cleanup" = {
      exec = "rm -rf build";
      after = [ "devenv:processes:serve" ];
    };
  };

  git-hooks.hooks = {
    nixpkgs-fmt.enable = true;
    ormolu.enable = true;
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

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  # scripts.hello.exec = ''
  #   echo hello from $GREET
  # '';

  # enterShell = ''
  #   hello
  #   git --version
  # '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  # enterTest = ''
  #   echo "Running tests"
  #   git --version | grep --color=auto "${pkgs.git.version}"
  # '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
