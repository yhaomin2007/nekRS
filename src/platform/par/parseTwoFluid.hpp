void parseTwoFluidSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (!ini->sections.count("two fluid")) {
    options.setArgs("TWO FLUID ENABLED", "FALSE");
    return;
  }

  options.setArgs("TWO FLUID ENABLED", "TRUE");
  options.setArgs("FLUID PRESSURE ELLIPTIC COEFF FIELD", "TRUE");
  options.setArgs("TWO FLUID ALPHA FLOOR", "1e-8");
  options.setArgs("TWO FLUID ALPHA DIFFUSIVITY", "0");
  options.setArgs("TWO FLUID GRAVITY X", "0");
  options.setArgs("TWO FLUID GRAVITY Y", "0");
  options.setArgs("TWO FLUID GRAVITY Z", "0");
  options.setArgs("TWO FLUID DRAG MULTIPLIER", "1");
  options.setArgs("TWO FLUID MIXTURE CONTINUITY TOLERANCE", "1e-9");
  options.setArgs("TWO FLUID COUPLING ITERATIONS", "2");
  options.setArgs("TWO FLUID PRESSURE CORRECTORS", "2");
  options.setArgs("TWO FLUID PRESSURE MAX ITERATIONS", "100");
  options.setArgs("TWO FLUID PRESSURE RESTART VECTORS", "30");
  options.setArgs("TWO FLUID PROJECTION ONLY", "FALSE");
  options.setArgs("TWO FLUID NATIVE ALPHA SCALAR", "FALSE");
  options.setArgs("TWO FLUID FREEZE ALPHA", "FALSE");

  // Dedicated scalar Helmholtz solver used for implicit alpha diffusion.
  // Start from the established velocity elliptic settings, then force the
  // scalar-compatible choices.
  std::vector<std::pair<std::string, std::string>> alphaSolverOptions;
  for (const auto &entry : options) {
    const std::string prefix = "FLUID VELOCITY";
    if (entry.first.find(prefix) == 0) {
      alphaSolverOptions.push_back(
          {"TWO FLUID ALPHA" + entry.first.substr(prefix.size()), entry.second});
    }
  }
  for (const auto &entry : alphaSolverOptions) options.setArgs(entry.first, entry.second);
  options.setArgs("TWO FLUID ALPHA SOLVER", "CG");
  options.setArgs("TWO FLUID ALPHA PRECONDITIONER", "JACOBI");
  options.setArgs("TWO FLUID ALPHA SOLVER TOLERANCE", "1e-10");
  options.setArgs("TWO FLUID ALPHA ELLIPTIC COEFF FIELD", "TRUE");

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
  extractReal("alphadiffusivity", "TWO FLUID ALPHA DIFFUSIVITY");
  extractReal("gravityx", "TWO FLUID GRAVITY X");
  extractReal("gravityy", "TWO FLUID GRAVITY Y");
  extractReal("gravityz", "TWO FLUID GRAVITY Z");
  extractReal("dragmultiplier", "TWO FLUID DRAG MULTIPLIER");
  extractReal("mixturecontinuitytolerance", "TWO FLUID MIXTURE CONTINUITY TOLERANCE");

  int couplingIterations;
  if (ini->extract("two fluid", "couplingiterations", couplingIterations))
    options.setArgs("TWO FLUID COUPLING ITERATIONS", std::to_string(couplingIterations));

  int pressureCorrectors;
  if (ini->extract("two fluid", "pressurecorrectors", pressureCorrectors))
    options.setArgs("TWO FLUID PRESSURE CORRECTORS", std::to_string(pressureCorrectors));

  int pressureMaxIterations;
  if (ini->extract("two fluid", "pressuremaxiterations", pressureMaxIterations))
    options.setArgs("TWO FLUID PRESSURE MAX ITERATIONS", std::to_string(pressureMaxIterations));

  int pressureRestartVectors;
  if (ini->extract("two fluid", "pressurerestartvectors", pressureRestartVectors))
    options.setArgs("TWO FLUID PRESSURE RESTART VECTORS", std::to_string(pressureRestartVectors));

  bool projectionOnly;
  if (ini->extract("two fluid", "projectiononly", projectionOnly))
    options.setArgs("TWO FLUID PROJECTION ONLY", projectionOnly ? "TRUE" : "FALSE");

  bool nativeAlphaScalar;
  if (ini->extract("two fluid", "nativealphascalar", nativeAlphaScalar))
    options.setArgs("TWO FLUID NATIVE ALPHA SCALAR",
                    nativeAlphaScalar ? "TRUE" : "FALSE");

  bool freezeAlpha;
  if (ini->extract("two fluid", "freezealpha", freezeAlpha))
    options.setArgs("TWO FLUID FREEZE ALPHA", freezeAlpha ? "TRUE" : "FALSE");

  // Reuse the liquid velocity solver settings for the gas phase.  The gas
  // boundary map can then be overridden independently so a liquid no-slip
  // wall does not also constrain the gas tangential velocity.
  std::vector<std::pair<std::string, std::string>> cloned;
  for (const auto &entry : options) {
    const std::string prefix = "FLUID VELOCITY";
    if (entry.first.find(prefix) == 0) {
      cloned.push_back({"GAS VELOCITY" + entry.first.substr(prefix.size()), entry.second});
    }
  }
  for (const auto &entry : cloned) options.setArgs(entry.first, entry.second);

  std::string gasBoundaryTypeMap;
  if (ini->extract("two fluid", "gasboundarytypemap", gasBoundaryTypeMap)) {
    options.setArgs("GAS VELOCITY BOUNDARY TYPE MAP", gasBoundaryTypeMap);
  }
  options.setArgs("GAS VELOCITY ELLIPTIC COEFF FIELD", "TRUE");
  options.setArgs("GAS CHECKPOINTING", "TRUE");
}
