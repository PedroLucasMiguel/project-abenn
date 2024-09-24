import os
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


def clear_terminal() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def generate_training_results_graphs(model: str) -> None:
    def generate_graphs(output_path: str) -> None:

        with open(os.path.join(output_path, "training_results.json"), "r") as f:
            metrics = json.load(f)

            n_epochs = len(metrics.keys())
            epochs_k = list(metrics.keys())
            metrics_per_epoch = len(metrics[list(metrics.keys())[0]].keys())
            metrics_k = list(metrics[epochs_k[0]].keys())

            np_metrics = np.zeros((metrics_per_epoch, n_epochs), dtype=np.float64)

            for i in range(metrics_per_epoch):
                for j in range(n_epochs):
                    np_metrics[i][j] = metrics[epochs_k[j]][metrics_k[i]]

            for metric in range(metrics_per_epoch):
                v = np_metrics[metric][:] * 100

                plt.gca().set_ylim(top=105)
                plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xlabel("Epochs")
                plt.xticks(range(1, n_epochs + 1))
                plt.ylabel(f"{metrics_k[metric].capitalize()} (%)")
                plt.plot(range(n_epochs), v)

                s_y = v[np.argmax(v)]
                s_x = np.argmax(v)
                plt.scatter(s_x, s_y)
                plt.annotate(f"{v[np.argmax(v)]:.2f}%",
                             (s_x + 0.2, s_y - 5 if s_y > 5 else s_y + 5),
                             weight="bold",
                             ha='center',
                             color=plt.gca().lines[-1].get_color())

                try:
                    os.mkdir(f"{output_path}/graphs")
                except OSError as _:
                    pass

                plt.tight_layout()
                plt.savefig(os.path.join(f"{output_path}/graphs", f"{metrics_k[metric]}.png"))
                plt.cla()

            plt.gca().set_ylim(top=105)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.xlabel("Epochs")
            plt.xticks(range(1, n_epochs + 1))

            last_x = -1
            last_y = -1
            p_increment = 0
            n_increment = 0

            for metric in range(metrics_per_epoch):
                v = np_metrics[metric][:] * 100
                plt.plot(v, label=metrics_k[metric].capitalize())

                s_y = v[np.argmax(v)]
                s_x = np.argmax(v)

                last_x = s_x if last_x == -1 else last_x

                text_y = s_y - 4.5 if s_y > 4.5 else s_y + 4.5
                text_x = s_x + 0.4

                if s_y == last_y or s_x == last_x:
                    last_x = s_x
                    last_y = s_y

                    if s_y > 5:
                        text_y = text_y - n_increment
                        n_increment += 5
                    else:
                        text_y = text_y + p_increment
                        p_increment += 5

                elif metric == 0:
                    last_x = s_x
                    last_y = s_y

                plt.scatter(s_x, s_y, color=plt.gca().lines[-1].get_color())
                plt.annotate(f"{v[np.argmax(v)]:.2f}%",
                             (text_x, text_y),
                             weight="bold",
                             ha='center',
                             color=plt.gca().lines[-1].get_color())

            plt.legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.15),
                       fancybox=True,
                       ncol=metrics_per_epoch)
            plt.tight_layout()
            plt.savefig(os.path.join(f"{output_path}/graphs", f"all_in_one.png"))
            plt.cla()

    datasets = os.listdir(f"../output/{model}")

    while True:
        clear_terminal()
        print("Select the output from one of those datasets: ")

        for dt_i in range(len(datasets)):
            print(f"[{dt_i}] - {datasets[dt_i]}")

        dt_i_i = int(input(">>> "))

        if dt_i_i < 0 or dt_i_i > len(datasets):
            print("Invalid Index")
            input("Press ENTER to continue...")

        else:
            clear_terminal()
            generate_graphs(f"../output/{model}/{datasets[dt_i_i]}")
            print(f"Finished!\nOutputs in: ../output/{model}/{datasets[dt_i_i]}/graphs")

            # TODO - Perguntar se o usuário deseja retornar ao menu ou não
            input("Press ENTER to return to menu...")
            break


def select_op(models: list[str]) -> None:
    while True:
        clear_terminal()
        print(f"Models: {models}\n")

        for m in models:
            print(f"Choose what to do with the {m} outputs:")
            print("[1] - Generate training results graphs")
            print("[2] - Compare CAM metrics with the second models")
            print("[3] - Compile CAM metris from this models")

            op = int(input(">>> "))

            match op:
                case 1:
                    generate_training_results_graphs(m)
                case 2:
                    pass
                case 3:
                    pass


def select_model_data(n_models: int = 1) -> list[str]:
    try:

        assert 1 <= n_models <= 2

        r = []

        models = os.listdir("output/")

        for i in range(n_models):
            clear_terminal()
            print(f"Select the {i + 1}° models")

            for m_i in range(len(models)):
                print(f"{m_i} - {models[m_i]}")

            # Kinda gross, but it will work for now
            m_i_i = int(input(">>> "))
            while m_i_i < 0 or m_i_i > len(models):
                print("\nERROR - Invalid models!")
                m_i_i = int(input(">>> "))

            r.append(models[m_i_i])

        return r

    except OSError as _:
        print("Compile Outputs - Output folder does not exist")

    except AssertionError as error:
        print(f"Compile Outputs - {error}")


def main_menu() -> None:
    select_op(select_model_data(1))

    pass


if __name__ == "__main__":
    main_menu()

    pass
