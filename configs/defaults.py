import re

splitext = lambda s: [x.replace("-", " ") for x in re.split("\\ +", s.replace("\n", " "))]


CACHE_DIR = ".cache"

TRAIN_SET = splitext(
    (
        """atlanta austin baltimore columbus dallas
        houston indianapolis london louisville dc
        milwaukee minneapolis nashville orlando sf
        philadelphia phoenix portland san-antonio
        san-jose seattle st-louis tampa vegas miami"""
    )
)
TEST_SET = splitext(
    (
        """montreal boston chicago denver kansas-city
        amsterdam new-york pittsburgh saltlakecity
        la paris san-diego tokyo toronto vancouver"""
    )
)
