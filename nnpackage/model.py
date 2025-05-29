from nnpackage.ctypes_bridger import load_lib

class CudaModel:
    def __init__(self, lib_path: str, model_path: str = None, use_cuda: bool = True):
        self.lib_funcs = load_lib(lib_path)
        if model_path is not None:
            self.model = self.lib_funcs.get("load")(model_path.encode())
        else:
            self.model = self.lib_funcs.get("create_model")(use_cuda)
            
    def details(self):
        self.lib_funcs.get("details")(self.model)
    
    def save(self, path: str, checkpoint: bool):
        self.lib_funcs.get("save")(self.model, path.encode(), checkpoint)