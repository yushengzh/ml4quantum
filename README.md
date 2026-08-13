

# Learning for Quantum Systems

Released code for the Paper *[Rethink the Role of Deep Learning towards Large-scale Quantum Systems](https://arxiv.org/pdf/2505.13852)*


## Dataset
The dataset we collected and used is released on [Google Drive](https://drive.google.com/drive/folders/1nxtzRjxHECQ3cXZh3pzCI5mN1izIK1th?usp=sharing). You can also generate your own quantum dataset by executing the `julia` scripts in path `ml4quantum/dataset_generation/.`, following the below steps.
### Setup
>
    git clone https://github.com/yushengzh/ml4quantum.git
    cd ml4quantum/dataset_generation

Launch Julia and activate the project.
- Julia (version = 1.11)
- Required packages include `Itensors.jl`, `ITensorMPS.jl`, `PastaQ.jl`.
>
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
This installs all necessary dependencies as defined in `Project.toml` and `Manifest.toml`.

### Example: Run with parameters
You can customize the dataset generation by modifying the following parameters in the script or passing them via the command line:
- `--samples_num` (`-n`): Number of Hamiltonian samples
- `--shots`(`-s`): Number of measurement shots per sample
- `--qubits` (`-q`): Number of qubits (Size of the Hamitonian system)
>
    julia generation_heisenberg_1d.jl -n 300 -s 1024 -q 8


<!--
To set up the environment, run the following command to install the other required packages listed in the requirements.txt file in the current directory:
>
    pip install -r requirements.txt
-->
## Citation

If you find the source code and datasets useful in your research, please cite:
>
    @inproceedings{zhao2025rethink,
    title={Rethink the Role of Deep Learning towards Large-scale Quantum Systems},
    author={Yusheng Zhao and Chi Zhang and Yuxuan Du},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
    }
