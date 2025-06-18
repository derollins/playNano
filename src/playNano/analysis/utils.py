import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing NumPy ndarray objects.

    This encoder converts NumPy arrays to native Python lists so they can be
    serialized by the standard `json` module. It can be used with `json.dump`
    or `json.dumps` by passing it as the `cls` argument.

    Example:
        json.dump(data, file, cls=NumpyEncoder)
    """

    def default(self, obj):
        """
        Override the default method to convert NumPy arrays to lists.

        Parameters:
            obj (Any): The object to be serialized.

        Returns:
            A JSON-serializable version of the object. If the object is a NumPy
            ndarray, it is converted to a list. Otherwise, the superclass's
            default method is used.

        Raises:
            TypeError: If the object cannot be serialized by the superclass.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
