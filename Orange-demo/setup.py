from setuptools import setup

setup(name="DemoSeminario",
      packages=["orangedemo"],
      package_data={"orangedemo": ["icons/*.svg", "icons/*.png"]},
      # Declare orangedemo package to contain widgets for the "Demo" category
      entry_points={"orange.widgets": "DemoSeminario = orangedemo"},
      )