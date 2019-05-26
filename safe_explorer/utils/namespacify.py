

class Namespacify(object):
    def __init__(self, name, in_dict):
        self.name = name

        for key in in_dict.keys():
            if isinstance(in_dict[key], dict):
                in_dict[key] = Namespacify(key, in_dict[key])
    
        self.__dict__.update(in_dict)

    def pprint(self, indent=0):
        print(f"{' ' * indent}{self.name}:")
        
        indent += 4
        
        for k,v in self.__dict__.items():
            if k == "name":
                continue
            if type(v) == Namespacify:
                v.pprint(indent)
            else:
                print(f"{' ' * indent}{k}: {v}")