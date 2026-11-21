/* contract.js -- what the API promised, and what it actually sent. */

const Contract = (function () {
  "use strict";

  let manifest = null;

  /** Load the manifest. Called once at start-up. */
  function install(loaded) {
    manifest = loaded;
  }

  /** Return the manifest, or an empty object before it has loaded. */
  function spec() {
    return manifest || {};
  }

  /**
   * Return the names in `names` that `object` does not carry. An explicit
   * `null` is an answer and counts as present.
   */
  function missing(object, names) {
    if (!object || typeof object !== "object") {
      return names.slice();
    }
    const absent = [];
    names.forEach(function (name) {
      if (!Object.prototype.hasOwnProperty.call(object, name)) {
        absent.push(name);
      }
    });
    return absent;
  }

  /** Check a whole `POST /api/run` response against the manifest. */
  function checkRun(payload) {
    const problems = [];
    const rules = spec();
    const top = (rules.endpoints && rules.endpoints.run) || {};

    missing(payload, top.required || []).forEach(function (name) {
      problems.push(`the response has no "${name}"`);
    });

    const detection = payload ? payload.detection : null;
    const detectionRules = rules.detection || {};
    missing(detection, detectionRules.required || []).forEach(function (name) {
      problems.push(`detection has no "${name}"`);
    });
    if (detection && detection.budget) {
      missing(detection.budget, detectionRules.budget_required || []).forEach(
        function (name) {
          problems.push(`detection.budget has no "${name}"`);
        }
      );
    }
    if (detection && Array.isArray(detection.signals)) {
      detection.signals.forEach(function (signal, index) {
        missing(signal, detectionRules.signal_required || []).forEach(
          function (name) {
            problems.push(`detection.signals[${index}] has no "${name}"`);
          }
        );
      });
    }
    if (detection && Array.isArray(detection.attributions)) {
      detection.attributions.forEach(function (row, index) {
        missing(row, detectionRules.attribution_required || []).forEach(
          function (name) {
            problems.push(`detection.attributions[${index}] has no "${name}"`);
          }
        );
      });
    }

    const runRules = rules.run_facts || {};
    missing(payload ? payload.run : null, runRules.required || []).forEach(
      function (name) {
        problems.push(`run has no "${name}"`);
      }
    );
    // The null banner picks one of THREE states off `run.nulls`, and it is the
    // panel that decides whether an honest run over a noisy link reads as an
    // attack.
    if (payload && payload.run && payload.run.nulls) {
      missing(payload.run.nulls, runRules.nulls_expected || []).forEach(
        function (name) {
          problems.push(`run.nulls has no "${name}"`);
        }
      );
    }

    const truthRules = rules.ground_truth || {};
    missing(
      payload ? payload.ground_truth : null,
      truthRules.required || []
    ).forEach(function (name) {
      problems.push(`ground_truth has no "${name}"`);
    });

    const timingRules = rules.timings || {};
    missing(payload ? payload.timings : null, timingRules.required || []).forEach(
      function (name) {
        problems.push(`timings has no "${name}"`);
      }
    );

    return { ok: problems.length === 0, problems: problems };
  }

  /** Read one field, or record that it was not supplied. */
  function field(object, name, path, report) {
    if (
      object &&
      typeof object === "object" &&
      Object.prototype.hasOwnProperty.call(object, name)
    ) {
      return object[name];
    }
    report.push(`${path}.${name}`);
    return undefined;
  }

  return {
    checkRun: checkRun,
    field: field,
    install: install,
    missing: missing,
    spec: spec,
  };
})();
