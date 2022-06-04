"""Defines Study data class and associated generator classes"""
import json
from abc import ABC, abstractmethod
import random
import madcad
import geometry as g
from stl import Mesh


class Study:
    """Data class representing a Moldflow study.

    Attributes:
        name: Name of the study.
        geometry: Part geometry.
        injection_locations: Positions, where the material gets injected into the geometry.
    """
    def __init__(self, name: str, geometry: Mesh, injection_locations: list) -> None:
        self.name = name
        self.geometry = geometry
        self.injection_locations = injection_locations


class StudyGenerator (ABC):
    """Abstract StudyGenerator. Concrete Implementations define the blueprint for the generation
        Study objects.
    """

    @abstractmethod
    def generate_study(self) -> Study:
        """Generates a study object."""
        pass


class PlateWithHoleGenerator (StudyGenerator):
    """ Generator that generates studies containing a part geometry of the form plate with a hole
        and a random injection location on the top of the geometry.

    Attributes:
        start_index: Start index for the naming of the studies generated. (default: 0)
    """

    CONFIG_FILE_PATH = "config_PlateWithHoleGenerator.json"

    def __init__(self, start_index: int = 0) -> None:
        super().__init__()
        self._study_index = start_index

        # load config file
        with open(PlateWithHoleGenerator.CONFIG_FILE_PATH, "r") as config_file:
            self.config = json.load(config_file)


    def generate_study(self) -> Study:
        """Generates a study object.

            Returns:
                A study object with a plate with a hole as part geometry and a random injection
                location on the top of the part geometry.
        """

        name = self.config["name_template"].format(study_index=self._study_index)
        self._study_index += 1

        # get random hole position
        max_hole_x = self.config["plate_width"] - 2 * self.config["hole_padding"] - 2 * self.config["hole_radius"]
        hole_x = self.config["hole_padding"] + self.config["hole_radius"] + random.randint(0, max_hole_x)
        max_hole_y = self.config["plate_height"] - 2 * self.config["hole_padding"] - 2 * self.config["hole_radius"]
        hole_y = self.config["hole_padding"] + self.config["hole_radius"] + random.randint(0, max_hole_y)

        # generate geometry
        plate = g.build_plate(self.config["plate_width"], self.config["plate_height"], self.config["plate_thickness"])
        hole = g.build_cylinder(hole_x, hole_y, self.config["hole_depth"], self.config["hole_radius"])
        geometry = madcad.difference(plate, hole)

        # get random injection location
        while True:
            inj_x = random.randint(0, self.config["plate_width"])
            inj_y = random.randint(0, self.config["plate_height"])

            if not g.is_on_circle(inj_x, inj_y, hole_x, hole_y, self.config["hole_radius"]):
                break

        inj_z = 0
        inj_locations = [((inj_x, inj_y, inj_z), tuple(self.config["injection_direction"]))]

        return Study(name, geometry, inj_locations)


# generator repository
generators = {
    "PlateWithHoleGenerator": PlateWithHoleGenerator()
}
