void parseTwoFluidSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (!ini->sections.count("two fluid")) {
    options.setArgs("TWO FLUID ENABLED", "FALSE");
    return;
  }

  options.setArgs("TWO FLUID ENABLED", "TRUE");
  options.setArgs("FLUID PRESSURE ELLIPTIC COEFF FIELD", "TRUE");
  options.setArgs("TWO FLUID ALPHA FLOOR", "1e-8");
  options.setArgs("TWO FLUID GRAVITY X", "0");
  options.setArgs("TWO FLUID GRAVITY Y", "0");
  options.setArgs("TWO FLUID GRAVITY Z", "0");
  options.setArgs("TWO FLUID DRAG MULTIPLIER", "1");
  options.setArgs("TWO FLUID COUPLING ITERATIONS", "2");
  options.setArgs("TWO FLUID PRESSURE CORRECTORS", "2");
  options.setArgs("TWO FLUID PROJECTION ONLY", "FALSE");

  auto extractReal = [&](const std::string &key, const std::string &option, bool required = false) {
    double value;
    if (ini->extract("two fluid", key, value)) {
      options.setArgs(option, to_string_f(value));
    } else if (required) {
      append_error("Missing TWO FLUID::" + key);
    }
  };

  extractReal("gasvolumefraction", "TWO FLUID GAS VOLUME FRACTION", true);
  extractReal("bubblediameter", "TWO FLUID BUBBLE DIAMETER", true);
  extractReal("gasdensity", "GAS DENSITY", true);
  extractReal("gasviscosity", "GAS VISCOSITY", true);
  extractReal("alphafloor", "TWO FLUID ALPHA FLOOR");
  extractReal("gravityx", "TWO FLUID GRAVITY X");
  extractReal("gravityy", "TWO FLUID GRAVITY Y");
  extractReal("gravityz", "TWO FLUID GRAVITY Z");
  extractReal("dragmultiplier", "TWO FLUID DRAG MULTIPLIER");

  int couplingIterations;
  if (ini->extract("two fluid", "couplingiterations", couplingIterations))
    options.setArgs("TWO FLUID COUPLING ITERATIONS", std::to_string(couplingIterations));

  int pressureCorrectors;
  if (ini->extract("two fluid", "pressurecorrectors", pressureCorrectors))
    options.setArgs("TWO FLUID PRESSURE CORRECTORS", std::to_string(pressureCorrectors));

  bool projectionOnly;
  if (ini->extract("two fluid", "projectiononly", projectionOnly))
    options.setArgs("TWO FLUID PROJECTION ONLY", projectionOnly ? "TRUE" : "FALSE");

  // v1 deliberately reuses the liquid velocity solver and boundary settings.
  // Clone every relevant option after the normal fluid parser has run.
  std::vector<std::pair<std::string, std::string>> cloned;
  for (const auto &entry : options) {
    const std::string prefix = "FLUID VELOCITY";
    if (entry.first.find(prefix) == 0) {
      cloned.push_back({"GAS VELOCITY" + entry.first.substr(prefix.size()), entry.second});
    }
  }
  for (const auto &entry : cloned) options.setArgs(entry.first, entry.second);
  options.setArgs("GAS VELOCITY ELLIPTIC COEFF FIELD", "TRUE");
  options.setArgs("GAS CHECKPOINTING", "TRUE");
}
