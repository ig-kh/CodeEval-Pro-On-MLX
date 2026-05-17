
def Strongest_Extension(class_name, extensions):
    """You will be given the name of a class (a string) and a list of extensions.
    The extensions are to be used to load additional classes to the class. The
    strength of the extension is as follows: Let CAP be the number of the uppercase
    letters in the extension's name, and let SM be the number of lowercase letters 
    in the extension's name, the strength is given by the fraction CAP - SM. 
    You should find the strongest extension and return a string in this 
    format: ClassName.StrongestExtensionName.
    If there are two or more extensions with the same strength, you should
    choose the one that comes first in the list.
    For example, if you are given "Slices" as the class and a list of the
    extensions: ['SErviNGSliCes', 'Cheese', 'StuFfed'] then you should
    return 'Slices.SErviNGSliCes' since 'SErviNGSliCes' is the strongest extension 
    (its strength is -1).
    Example:
    for Strongest_Extension('my_class', ['AA', 'Be', 'CC']) == 'my_class.AA'
    """
    strongest = None
    max_strength = float('-inf')
    
    for ext in extensions:
        cap = sum(1 for char in ext if char.isupper())
        sm = sum(1 for char in ext if char.islower())
        strength = cap - sm
        
        if strength > max_strength:
            max_strength = strength
            strongest = ext
            
    return f"{class_name}.{strongest}" if strongest else f"{class_name}.None"

def Strongest_Extensions(classes_with_extensions):
    """Given a list of classes with their respective extensions, determine the strongest extension for each class and return a list of strings in the format: ClassName.StrongestExtensionName. If a class has no extensions, it should be represented as ClassName.None. The strength of an extension is calculated as the difference between the number of uppercase letters and the number of lowercase letters in the extension's name. If there are two or more extensions with the instruction.
    """
    result = []
    for class_name, extensions in classes_with_extensions:
        result.append(Strongest_Extension(class_name, extensions))
    return result


assert Strongest_Extensions({'Slices': ['SErviNGSliCes', 'Cheese', 'StuFfed'], 'my_class': ['AA', 'Be', 'CC']}) == ['Slices.SErviNGSliCes', 'my_class.AA']
assert Strongest_Extensions({'Empty': [], 'Single': ['A']}) == ['Empty.None', 'Single.A']
assert Strongest_Extensions({'Mixed': ['aA', 'Bb', 'Cc'], 'AllCaps': ['AA', 'BB', 'CC']}) == ['Mixed.aA', 'AllCaps.AA']
assert Strongest_Extensions({'NoExtensions': []}) == ['NoExtensions.None']
assert Strongest_Extensions({}) == []  # Check for empty input
