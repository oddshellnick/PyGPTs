class ObjectTypeKwargs:
    """
    A base class for storing keyword arguments to be passed to other object constructors.

    :Usage:
        from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs

        kwargs = ObjectTypeKwargs(name="example", value=123)
        print(kwargs.name)  # Output: example
        print(kwargs.value)  # Output: 123
    """

    def __init__(self, **kwargs):
        """
        Initializes ObjectTypeKwargs by setting attributes based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to be stored as attributes.
        """
        for name, value in kwargs.items():
            setattr(self, name, value)
