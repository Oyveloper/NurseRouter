import json
import os
import glob
import matplotlib.pyplot as plt
from numpy import sort


def main():
    colors = [
        "lightcoral",
        "chocolate",
        "yellow",
        "olivedrab",
        "palegreen",
        "aquamarine",
        "cadetblue",
        "pink",
        "indigo",
        "plum",
        "peachpuff",
        "beige",
        "black",
        "lightgray",
        "moccasin",
        "teal",
    ]

    files = glob.glob("images/plot_data/*.json")
    file_name = sorted(files, key=os.path.getctime)[-3]
    problem = file_name.split("-")[1]
    print(file_name)
    with open(file_name) as f:
        solution = json.load(f)

    with open("assets/instances/train_" + problem + ".json") as f:
        problem = json.load(f)

    # Plot the depot
    depot = problem["depot"]

    # plot each route

    patients = problem["patients"]
    i = 0
    for route in solution["routes"]:
        locations = ["depot"] + route + ["depot"]

        last_location = None
        for location in locations:

            coordinates = depot if location == "depot" else patients[str(location)]
            x = coordinates["x_coord"]
            y = coordinates["y_coord"]

            if location != "depot":
                plt.plot(x, y, "ko")

            if last_location is not None:
                plt.plot(
                    [last_location["x_coord"], x],
                    [last_location["y_coord"], y],
                    "-",
                    color=colors[i],
                )

            last_location = coordinates
        i = (i + 1) % len(colors)

    plt.plot(depot["x_coord"], depot["y_coord"], "ks")

    plt.show()


if __name__ == "__main__":
    main()
