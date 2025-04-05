Module MolGenX-backend.RnnClass
===============================

Functions
---------

`generate_diverse_molecules(model: torch.nn.modules.module.Module, char_to_idx: dict, idx_to_char: dict, device: torch.device, start_token: str = 'C', num_molecules: int = 10, max_length: int = 100, max_attempts: int = 100) ‑> List[str]`
:   Generate multiple unique and valid molecules
    
    Args:
        model (nn.Module): Trained RNN model
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        device (torch.device): Device to run model on
        start_token (str): Starting token for molecule generation
        num_molecules (int): Number of unique molecules to generate
        max_length (int): Maximum length of molecule SMILES
        max_attempts (int): Maximum generation attempts
    
    Returns:
        List[str]: List of unique valid molecules

Classes
-------

`RNNGenerator(vocab_size, embed_dim, hidden_dim)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.