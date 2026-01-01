{ pkgs
, lib
, config
, inputs
, ...
}:

let
  fsatrace = pkgs.stdenv.mkDerivation {
    pname = "fsatrace";
    version = "0-unstable-2025-09-12";

    src = pkgs.fetchFromGitHub {
      owner = "jacereda";
      repo = "fsatrace";
      rev = "4d4a967293eed5bd2a0298c5be6858e3f7fccb28";
      sha256 = "sha256-JSj9iyOAK1KEUn2pwEGHzAlkkquUtq57UmTJxwazsmM=";
    };

    nativeBuildInputs =
      [ pkgs.gnumake ]
      ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.clang pkgs.darwin.cctools ];

    buildPhase = "make";

    installPhase = ''
      mkdir -p "$out/bin" "$out/lib"
      # main tracer
      install -Dm755 ./fsatrace "$out/bin/fsatrace"

      # injector(s): keep next to the binary and also in lib
      if [ -f ./fsatrace.so ]; then
        install -Dm755 ./fsatrace.so "$out/bin/fsatrace.so"
        install -Dm755 ./fsatrace.so "$out/lib/fsatrace.so"
      fi

      for d in *.dylib; do
        if [ -f "$d" ]; then
          install -Dm755 "./$d" "$out/bin/$d"
          install -Dm755 "./$d" "$out/lib/$d"
        fi
      done
    '';

    meta = with pkgs.lib; {
      description = "Filesystem access tracer (needed by Shake Forward)";
      license = licenses.isc;
      platforms = platforms.unix;
    };
  };
in
{
  cachix.push = "tscholak";

  claude.code.enable = true;

  packages = [
    pkgs.git
    pkgs.nodejs
    pkgs.nodePackages.serve
    pkgs.watchexec
    pkgs.check-jsonschema
    fsatrace
  ];

  languages.haskell = {
    enable = true;
    package = pkgs.haskell.compiler.ghc910;
    stack.enable = true;
    languageServer = lib.mkIf (config.devenv.isTesting) null;
  };
  languages.nix.enable = true;
  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    venv.requirements = ''
      rendercv[full]
    '';
  };

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
    "build-resume-pdf" = {
      description = "Build resume PDF with rendercv";
      exec = ''
        EMAIL=`ghc -e "email" site/Contact.lhs 2>/dev/null | tr -d '"'`
        TMPDIR=`mktemp -d`
        TMPFILE="$TMPDIR/resume.yaml"
        sed "s/^  name: .*/&\n  email: \"$EMAIL\"/" site/resume.yaml > "$TMPFILE"
        rendercv render "$TMPFILE"
        mkdir -p build/resume
        cp "$TMPDIR/rendercv_output/"* build/resume/
        rm -rf "$TMPDIR"
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
    rendercv-schema = {
      enable = true;
      name = "RenderCV schema";
      entry = "${pkgs.check-jsonschema}/bin/check-jsonschema --schemafile https://raw.githubusercontent.com/rendercv/rendercv/refs/tags/v2.6/schema.json";
      files = "^site/resume\\.yaml$";
      language = "system";
    };
  };
}
