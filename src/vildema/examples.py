from vildema.database import Database

if __name__ == '__main__':
  db = Database()
  db.load_from_disk(paths=[("runs/**/*.json", "json")])

  # todo: we need to write the examples, borrowing them from the two projects where the baracca is already used