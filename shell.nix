{ config ? {}
, sourcesOverride ? {}
, pkgs
, tokenizers-haskell
}:
with pkgs;
let
  shell = pkgs.website-project.shellFor {
    name = "website-dev-shell";
    tools = {
      cabal = "latest";
      haskell-language-server = "latest";
    };
    exactDeps = true;
    withHoogle = true;
  };
in
  shell