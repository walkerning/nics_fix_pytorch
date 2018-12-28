## Fixed Point Training Simulation Framework on PyTorch

### Core Functionality
- Parameter fix: by using `nics_fix_pt.nn_fix.<original name>_fix` modules
- Activation fix: by using `ActivationFix` module
- Data fix VS. Gradient fix: by supply `nf_fix_params`/`nf_fix_params_grad` kwargs args
      in `nics_fix_pt.nn_fix.<original name>_fix` or `ActivationFix` module construction
- [ ] Handle buffers, such as bn running_mean/running_var

### Utilities
- [ ] FixTopModule: dump/load fix configuration to/from file

### Examples
See `examples/mnist/train_mnist.py` for a MNIST fix-point training example.

### Test cases

Run `python setup.py test` to run the pytest test cases.