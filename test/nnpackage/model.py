from nnpackage.ctypes_bridger import load_lib

class CudaModel:
    def __init__(self, lib_path: str, model_path: str = None):
        self.lib_funcs = load_lib(lib_path)
        if model_path is not None:
            self.model = self.lib_funcs.get("load")(model_path.encode())
        else:
            self.model = self.lib_funcs.get("create_model")()
            
    def get_details(self):
        self.lib_funcs.get("details")(self.model)
    
    def save_weights(self, path: str):
        self.lib_funcs.get("save")(self.model, path.encode())

    def load_weights(self, path: str):
        self.lib_funcs.get("load")(self.model, path.encode())
    
    def delete_memory(self):
        self.lib_funcs.get("delete")(self.model)