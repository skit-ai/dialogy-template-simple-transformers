"""
Import class

- ProductEntity

To read more about creating custom entities visit:
https://vernacular-ai.github.io/dialogy/dialogy/types/entity/base_entity.html
"""
from typing import List, Dict

import attr
from dialogy.types.entity import BaseEntity


@attr.s
class ProductEntity(BaseEntity):
    """
    Extend BaseEntity for products of any kind.

    An instance of this class should be used to
    wrap tokens that are understood as product entities.
    """
    values = attr.ib(
        type=List[Dict[str, str]],
        default=attr.Factory(list),
        validator=attr.validators.instance_of(List),
    )
    _meta = attr.ib(type=Dict[str, str], default=attr.Factory(dict))
