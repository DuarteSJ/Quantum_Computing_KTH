{
  description = "Python development environment";
  
  inputs = {
    nixpkgs.url = "github:DuarteSJ/nixpkgs/current";
  };
  
  outputs = { self, nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
      python = pkgs.python3;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          (python.withPackages (p: [
            p.numpy
            p.matplotlib
            p.qiskit
            p.qiskit-aer
            p.cirq
            p.pytest
            p.black
          ]))
        ];
        shellHook = ''
          echo -e "\n\033[1;36m🐍 Python development shell activated!\033[0m"
          echo -e "\033[0;90m    → Virtual environment: (py-env)\033[0m"
          
          export NIX_PS1_OVERRIDE="(py-env) "
          export PYTHONPATH="$(git rev-parse --show-toplevel)/quantum-simulator:$PYTHONPATH"
          exec zsh
        '';
      };
    };
}
