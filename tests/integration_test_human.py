from vildema import database

from vildema.database import C, Query
from vildema import ExperimentData
from vildema.tui import Browser, Visualizer
# import Path
from pathlib import Path


def render_samples(entry: ExperimentData) -> Path:
  """Return a bitmap for an entry's generated samples.

  A visualizer hands back pixels — here a path to the samples PNG, but a numpy
  array or PIL image (e.g. ``tensor.cpu().numpy()``) works just as well. The
  Browser previews it in-terminal with integer scaling; press `v`, then q/esc.
  """
  rel = entry[IMG_KEY]
  return Path(ROOT + str(rel)).expanduser()


db = database.Database()
db.load_from_disk(
  paths=[
    ("/home/dario/Phd/MMD_Flow/experiments/**/record_final.json", "json")
  ]
)

print(f"Found {len(db.entries)} entries")


q = Query()
# q.add_filter(C("params//feature_map//classifier") == 'lenet')
# q.add_filter(C("run_name").is_in(['lenet_mnist_fedkseed_20240617_173000']))
# q.add_filter(C("quality//judge_acc", 'skip') != None)
q.add_filter(C("params//data//dataset") == 'cifar10')
q.add_filter(C("params//data//class_filter") == None)
res = q.apply(db)

print(f"Found {len(res.entries)} entries after filtering")

IMG_KEY = 'artifacts//samples_image'
ROOT = "/home/dario/Phd/MMD_Flow/"

if False:
  from pathlib import Path
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  image_paths = []

  stop = False
  for i, entry in enumerate(res.entries):
    img_path = entry[IMG_KEY]
    if not img_path:
      continue

    img_path = ROOT + img_path

    path = Path(img_path).expanduser()
    if path.exists():
      image_paths.append(path)

    # print(f"Found {len(image_paths)} existing image files")
    #
    #
    # for i, path in enumerate(image_paths, start=1):
    if stop:
      break

    img = mpimg.imread(path)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
      f"{i}/{len(res.entries)}: {path.name}\nJudge accuracy={entry["quality//judge_acc"]}\nPress q = next, x = exit")


    def on_key(event):
      nonlocal_stop = False

      if event.key == "q":
        plt.close(fig)

      elif event.key == "x":
        global stop
        stop = True
        plt.close(fig)


    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
# ---------------------------------------------------------------------------
# TUI browser example.
#
# This shows the full workflow: build a Browser, register project-specific
# columns / formatters / keybinds / a visualizer, then run it. In a real
# project `render_samples` would live in its own module and be imported here
# (see cli_ui_specification.md §6) -- it is inline only to keep this one file
# self-contained.
# ---------------------------------------------------------------------------

# `column=IMG_KEY` scopes the key: `v` only opens the image while the samples
# column is focused, instead of firing from any cell in the row.
samples = Visualizer(key="v", name="samples-image", render=render_samples, column=IMG_KEY)

browser = Browser(title="MMD-Flow runs")
(browser.attach_db(res)
  .format("metrics//mmd_sq", lambda v: f"{v:.3e}")
  .sort_default("quality//judge_acc", descending=True)
  .add_visualizer(samples)  # press `v` on a row to open its sample image
  .hide_constant_columns()
  .hide_columns(["run_identifier", "timestamp"])
  # .hide_columns(["params//run//run_identifier", "params//run//out_dir", "params//run//started_at"])
  .hide_columns(["params//run"])
  .hide_columns(["artifacts//checkpoint"])
 )
browser.run()
