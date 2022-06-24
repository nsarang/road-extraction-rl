HRNET_CFG = dict(
    litehrnet18=dict(
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=("LITE", "LITE", "LITE"),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                ),
            ),
            with_head=True,
        ),
    ),
    litehrnet30=dict(
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=("LITE", "LITE", "LITE"),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                ),
            ),
            with_head=True,
        ),
    ),
)

DDRNET_CFG = dict(
    DDRNet23S=dict(layers=[2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=True),
    DDRNet39=dict(layers=[3, 4, 6, 3], num_classes=19, planes=64, spp_planes=128, head_planes=256, augment=True),
)
