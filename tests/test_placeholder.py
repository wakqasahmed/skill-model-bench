"""Placeholder so pytest and CI have something to run before real modules land."""


def test_package_imports():
    import skill_model_bench

    assert skill_model_bench.__version__ == "0.1.0"
