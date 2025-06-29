{
  description = "Flake used for computational science. Shows how to add python packages to config when unavailable.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    # old_nixpkgs.url = "github:nixos/nixpkgs/194c2aa446b2b059886bb68be15ef6736d5a8c31";
  };

  outputs = {
    self,
    nixpkgs,
    # old_nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    # oldpkgs = old_nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs; [
        uv
        pre-commit
      ];
    };

    doCheck = false;
  };
}
