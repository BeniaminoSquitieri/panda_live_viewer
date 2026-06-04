from perception.segmenters import SegmenterConfig, build_segmenter, labels_from_registry


def test_labels_from_registry_includes_canonical_names_and_aliases():
    labels = labels_from_registry(
        {
            "objects": [
                {"canonical_name": "toast", "aliases": ["bread"]},
                {"canonical_name": "cup", "aliases": ["mug"]},
            ]
        }
    )

    # OWL-ViT is fed the natural-language aliases AND the canonical names
    # (sorted), because the zero-shot detector handles aliases better;
    # canonicalize() maps every alias back to its canonical name.
    assert labels == ["bread", "cup", "mug", "toast"]


def test_build_noop_segmenter_does_not_load_model():
    segmenter = build_segmenter(
        SegmenterConfig(
            backend="noop",
            model_path="unused",
            score_threshold=0.5,
            mask_mode="box",
            image_color_order="bgr",
        ),
        registry={"objects": ["toast"]},
    )

    assert list(segmenter.detect(None)) == []
