class RomanPipeline:
    """
    Coordinates the complete Roman exoplanet workflow.
    """

    def __init__(
        self,
        roman_file: str,
        model_file: str,
        output_dir: str = "output",
    ):

        self.roman_file = roman_file
        self.model_file = model_file
        self.output_dir = Path(output_dir)

        # Bootstrap resources

        bootstrap = ProjectBootstrap()

        bootstrap.ensure_resources(

            self.roman_file,

            self.model_file

        )