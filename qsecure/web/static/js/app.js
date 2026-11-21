/* app.js: controls, transport, routing, theme, and the two modes. */

const App = (function () {
  "use strict";

  const h = Render.h;

  /** Where the service mounts the frontend's own files. */
  const STATIC = "/static";

  /** Where the chosen theme is remembered, per browser. */
  const THEME_KEY = "qsecure-theme";

  const state = {
    mode: "unknown",
    defaults: null,
    attacks: null,
    constants: null,
    docs: null,
    recorded: null,
    // file name -> the recorded `POST /api/run` response, held from boot so a
    // click never needs the network. See the header.
    recordedPayloads: {},
    recordedHeld: 0,
    busy: false,
    // The run every run-scoped view is about, and how to introduce it.
    payload: null,
    payloadTitle: null,
    payloadWhy: null,
    payloadSource: null,
    selectedFile: null,
    // True for exactly one render after a run is chosen: the replay plays once
    // per selection, not every time someone comes back to the session view.
    autoplay: false,
  };

  /** The bundle every view is rendered against. */
  function context() {
    return {
      defaults: state.defaults,
      attacks: state.attacks,
      constants: state.constants,
    };
  }

  /* -- Transport -- */

  /**
   * Fetch JSON from a same-origin path. Nothing here ever leaves the origin.
   */
  function getJson(path, options) {
    return fetch(path, options).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) {
          const detail = body && body.message ? body.message : "";
          throw new Error(`HTTP ${response.status} ${detail}`);
        }
        return body;
      });
    });
  }

  /* -- The roster -- */

  /** How each roster entry is introduced: never a verdict colour. */
  const KIND = {
    "not-an-attack": { word: "Baseline", css: "is-baseline" },
    detectable: { word: "Attack", css: "is-attack" },
    "undetectable-by-construction": {
      word: "Undetectable by design",
      css: "is-hidden",
    },
  };

  /** One roster entry by key, or null. */
  function rosterEntry(key) {
    let found = null;
    (state.attacks || []).forEach(function (entry) {
      if (found === null && entry.key === key) {
        found = entry;
      }
    });
    return found;
  }

  /* -- Run a session -- */

  /** Return one field's cap block from `/api/defaults`, or an empty object. */
  function capsFor(name) {
    const limits = (state.defaults && state.defaults.limits) || {};
    const fields = limits.fields || {};
    return fields[name] || {};
  }

  /** Return the API's own starting value for one request field. */
  function startingValue(name) {
    const params = (state.defaults && state.defaults.params) || {};
    if (!Object.prototype.hasOwnProperty.call(params, name)) {
      return "";
    }
    return String(params[name]);
  }

  /** One numeric setting, with the API's own range beside it. */
  function numberField(id, label, step, hint) {
    const caps = capsFor(id);
    const attrs = { type: "number", value: startingValue(id), step: step };
    if (caps.minimum !== undefined) {
      attrs.min = String(caps.minimum);
    }
    if (caps.maximum !== undefined) {
      attrs.max = String(caps.maximum);
    }
    const input = h("input", { attrs: attrs });
    input.id = id;
    const range =
      caps.minimum === undefined
        ? "The API didn't supply a range"
        : `${caps.minimum} to ${caps.maximum}`;
    return h("div", { class: "field" }, [
      h("label", { text: label, attrs: { for: id } }),
      input,
      h("span", { class: "field-hint", text: range }),
      hint ? h("span", { class: "field-hint", text: hint }) : null,
    ]);
  }

  /** One titled block of the Run view. */
  function runSection(title, children) {
    return h(
      "section",
      { class: "panel-card" },
      [h("h2", { class: "panel-title", text: title })].concat(children)
    );
  }

  /** The Run a session view. */
  function controls() {
    const detail = h("p", { class: "adversary-detail" });
    const grid = h("div", {
      class: "adversary-grid",
      attrs: { role: "radiogroup", "aria-label": "Adversary" },
    });
    const chosen = startingValue("attack");
    let anyChecked = false;
    (state.attacks || []).forEach(function (entry) {
      const kind = KIND[entry.detectable] || { word: "", css: "" };
      const radio = h("input", {
        attrs: { type: "radio", name: "attack", value: entry.key },
      });
      if (entry.key === chosen) {
        radio.checked = true;
        anyChecked = true;
        detail.textContent = Fmt.prose(entry.summary);
      }
      radio.addEventListener("change", function () {
        detail.textContent = Fmt.prose(entry.summary);
      });
      grid.appendChild(
        h("label", { class: "adversary-tile" }, [
          radio,
          h("span", { class: "tile-name", text: entry.label }),
          h("span", { class: `tile-kind ${kind.css}`, text: kind.word }),
        ])
      );
    });
    if (!anyChecked) {
      const first = grid.querySelector('input[name="attack"]');
      if (first) {
        first.checked = true;
      }
    }

    // The roster comes from the API's own limits block where it publishes
    // one, so a third ordering added upstream appears here without an edit.
    const timings = capsFor("count_exchange_timing").values || [
      "before-forwarding",
      "after-forwarding",
    ];
    const timingSelect = h("select", {});
    timingSelect.id = "count_exchange_timing";
    timings.forEach(function (value) {
      const option = h("option", { text: value, attrs: { value: value } });
      if (value === startingValue("count_exchange_timing")) {
        option.selected = true;
      }
      timingSelect.appendChild(option);
    });

    const runButton = h("button", {
      class: "run-button",
      text: "Run session",
      attrs: { type: "button", id: "run" },
    });
    runButton.addEventListener("click", startLiveRun);

    const limits = (state.defaults && state.defaults.limits) || {};

    return h("div", { class: "run-view" }, [
      h("header", { class: "view-head" }, [
        h("div", { class: "view-titles" }, [
          h("h1", { text: "Run a session" }),
          h("p", {
            class: "view-lede",
            text:
              "Pick an adversary and settings, and the server simulates a " +
              "new session for the detector to score, usually within " +
              "seconds. The API refuses a setting outside its range instead " +
              "of quietly adjusting it.",
          }),
        ]),
      ]),
      state.mode === "live"
        ? null
        : h("p", {
            class: "offline-note",
            text:
              `The live service isn't answering, so no session can run ` +
              `right now. ${fallbackNote()}`,
          }),
      runSection("Adversary", [grid, detail]),
      runSection("Settings", [
        h("div", { class: "field-row" }, [
          numberField("key_length", "Key length", "1"),
          numberField(
            "check_fraction",
            "Share of positions spent checking the link",
            "0.05"
          ),
          numberField(
            "noise",
            "Noise on the link",
            "0.005",
            "With noise above zero, set both noise assumptions under " +
              "Advanced settings, or an honest run can still raise an alarm."
          ),
        ]),
        h("details", { class: "advanced" }, [
          h("summary", { text: "Advanced settings" }),
          // The two nulls are ONE instruction and are labelled as one.
          h("p", {
            class: "warn-note",
            text:
              "TWO NULLS, AND BOTH DEFAULT TO A PERFECT LINK. detect() reads " +
              "the verifiers' mismatch counts against channel_error_rate and " +
              "the published check rounds against tolerated_depolarising. To " +
              "score an honest run over a noisy link against the link, set " +
              "both: setting either alone leaves the other family scoring " +
              "against a link nobody has, and the run can still fire. Nothing " +
              "infers either null from the transcript.",
          }),
          h("div", { class: "field-row" }, [
            numberField(
              "channel_error_rate",
              "NULL 1 of 2: channel_error_rate (rate family)",
              "0.005"
            ),
            numberField(
              "tolerated_depolarising",
              "NULL 2 of 2: tolerated_depolarising (channel family)",
              "0.005"
            ),
          ]),
          h("div", { class: "field-row" }, [
            numberField("eps", "Alarm budget, eps", "any"),
            h("div", { class: "field" }, [
              h("label", {
                text: "When Bob and Charlie exchange counts",
                attrs: { for: "count_exchange_timing" },
              }),
              timingSelect,
              h("span", {
                class: "field-hint",
                text:
                  "Against a forging recipient the two orders answer " +
                  "different questions: before forwarding the attack is a " +
                  "denial of transfer, and after forwarding it's a forgery " +
                  "Charlie scores.",
              }),
            ]),
            numberField("seed", "Random seed", "1"),
          ]),
        ]),
      ]),
      h("div", { class: "run-row" }, [
        runButton,
        h("div", {
          class: "run-status",
          text: "Ready",
          attrs: { id: "run-status", role: "status", "aria-live": "polite" },
        }),
      ]),
      limits.max_concurrent_runs === undefined
        ? null
        : h("p", {
            class: "field-hint",
            text:
              `The server runs ${limits.max_concurrent_runs} ${
                limits.max_concurrent_runs === 1 ? "session" : "sessions"
              } at a time and refuses the rest straight away instead of ` +
              `queueing them.`,
          }),
    ]);
  }

  /** Read the controls. No conversion beyond the DOM's own `valueAsNumber`. */
  function readControls() {
    const picked = document.querySelector('input[name="attack"]:checked');
    return {
      attack: picked ? picked.value : "honest",
      key_length: document.getElementById("key_length").valueAsNumber,
      check_fraction: document.getElementById("check_fraction").valueAsNumber,
      noise: document.getElementById("noise").valueAsNumber,
      eps: document.getElementById("eps").valueAsNumber,
      channel_error_rate: document.getElementById("channel_error_rate")
        .valueAsNumber,
      tolerated_depolarising: document.getElementById("tolerated_depolarising")
        .valueAsNumber,
      count_exchange_timing: document.getElementById("count_exchange_timing")
        .value,
      seed: document.getElementById("seed").valueAsNumber,
    };
  }

  /* -- Recorded runs -- */

  /** Fetch every recorded run into memory, once, at boot. */
  function preloadRecorded() {
    const entries = state.recorded || [];
    return Promise.all(
      entries.map(function (entry) {
        return getJson(`${STATIC}/data/recorded/${entry.file}`)
          .then(function (payload) {
            state.recordedPayloads[entry.file] = payload;
          })
          .catch(function () {
            /* Not held. `recordedHeld` below counts what actually is. */
          });
      })
    ).then(function () {
      state.recordedHeld = Object.keys(state.recordedPayloads).length;
    });
  }

  /** The scenario rail: every recorded run, one click each, in two groups. */
  function recordedList() {
    const rail = h("div", { class: "scenario-rail" });
    const honestList = h("ul", { class: "scenario-list" });
    const attackList = h("ul", { class: "scenario-list" });
    let anyHonest = false;
    let anyAttack = false;
    (state.recorded || []).forEach(function (entry) {
      const held = Object.prototype.hasOwnProperty.call(
        state.recordedPayloads,
        entry.file
      );
      const roster = rosterEntry(entry.attack);
      const attack = roster !== null && roster.detectable !== "not-an-attack";
      const current = state.selectedFile === entry.file;
      const button = h("button", {
        class: `scenario${current ? " is-current" : ""}${
          attack ? " is-attack" : ""
        }${held ? "" : " is-missing"}`,
        attrs: {
          type: "button",
          title: Fmt.prose(entry.why),
          "aria-pressed": current ? "true" : "false",
          "aria-label": `${entry.label}${attack ? ", attack scenario" : ""}${
            held ? "" : ", not loaded"
          }`,
        },
      }, [
        attack
          ? h("span", { class: "scenario-mark", attrs: { "aria-hidden": "true" } })
          : null,
        h("span", { class: "scenario-name", text: entry.label }),
      ]);
      button.addEventListener("click", function () {
        showRecorded(entry);
      });
      if (attack) {
        attackList.appendChild(h("li", {}, [button]));
        anyAttack = true;
      } else {
        honestList.appendChild(h("li", {}, [button]));
        anyHonest = true;
      }
    });
    if (anyHonest) {
      rail.appendChild(
        h("section", { class: "rail-group" }, [
          h("h2", { class: "rail-heading", text: "Honest runs" }),
          honestList,
        ])
      );
    }
    if (anyAttack) {
      rail.appendChild(
        h("section", { class: "rail-group" }, [
          h("h2", { class: "rail-heading", text: "Attacks" }),
          attackList,
        ])
      );
    }
    const total = (state.recorded || []).length;
    rail.appendChild(
      h("p", {
        class:
          total > 0 && state.recordedHeld === total
            ? "held-note"
            : "held-note is-short",
        text:
          total === 0
            ? "No recorded scenarios: the service that serves them didn't " +
              "answer. Start the server and reload."
            : `${state.recordedHeld} of ${total} held in this page's memory, ` +
              `so they keep working if the service stops.`,
      })
    );
    return rail;
  }

  /** Show one recorded run, from memory. */
  function showRecorded(entry) {
    const payload = state.recordedPayloads[entry.file];
    if (!payload) {
      // Never a silent blank and never the previous run left standing.
      noteTransportFailure(
        `This page didn't load ${entry.file} into memory at start-up, and it ` +
          `never fetches a recorded run on click. If the service is running, ` +
          `reload the page to pick it up.`,
        { recorded_run: entry.file }
      );
      setStatus(`Recorded run not held: ${entry.file}`, "failed");
      return;
    }
    hold(payload, {
      title: entry.label,
      why: entry.why,
      source: "Recorded run",
      file: entry.file,
    });
    setStatus(`Showing the recorded run ${entry.label}, from memory`, "");
    go("session");
  }

  /** The two runs Home plays on a loop, honest first, only if both are held. */
  function homeReel() {
    const reel = [];
    ["honest", "outside_forgery"].forEach(function (scenario) {
      (state.recorded || []).forEach(function (entry) {
        const payload = state.recordedPayloads[entry.file];
        if (entry.scenario === scenario && payload) {
          reel.push({
            key: entry.scenario,
            title: entry.label,
            why: entry.why,
            file: entry.file,
            payload: payload,
          });
        }
      });
    });
    return reel;
  }

  /** Open one Home reel run in the Session view. */
  function openReelItem(item) {
    showRecorded({ file: item.file, label: item.title, why: item.why });
  }

  /* -- Live runs -- */

  /** What still works when the service doesn't answer. */
  function fallbackNote() {
    return state.recordedHeld > 0
      ? "Recorded scenarios held in this page's memory still work."
      : "This page holds no recorded scenario either, so start the server " +
          "and reload.";
  }

  /** Set the status line, where the Run view is showing one. */
  function setStatus(text, cls) {
    const node = document.getElementById("run-status");
    if (!node) {
      return;
    }
    node.className = `run-status ${cls}`;
    node.textContent = text;
  }

  /** Report that nothing answered, in the view AND in the top bar. */
  function noteTransportFailure(message, request) {
    clearHeld();
    Render.refused(
      document.getElementById("view"),
      message,
      request,
      "unreachable"
    );
    state.mode = "recorded";
    paintMode();
  }

  /**
   * Start one live run. One at a time: a second click is refused rather than
   * queued, because two multi-second sessions racing is how a demo hangs.
   */
  function startLiveRun() {
    if (state.busy) {
      return;
    }
    if (state.mode !== "live") {
      // A click that started no run must not leave the previous run's verdict
      // standing.
      noteTransportFailure(
        `The API isn't reachable, so no live run started. ${fallbackNote()}`,
        readControls()
      );
      setStatus(
        `The API isn't reachable, so no session can run. ${fallbackNote()}`,
        "failed"
      );
      return;
    }
    const body = readControls();
    const button = document.getElementById("run");
    state.busy = true;
    button.disabled = true;
    setStatus(
      `Running ${body.attack} at key length ${body.key_length}. Generating ` +
        `the session is the slow part; detection takes milliseconds.`,
      "busy"
    );
    getJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (payload) {
        const roster = rosterEntry(body.attack);
        hold(payload, {
          title: roster ? roster.label : body.attack,
          why: "A session generated just now, with the settings you chose.",
          source: "Live run",
          file: null,
        });
        setStatus("Session complete", "");
        go("session");
      })
      .catch(function (error) {
        // `getJson` throws "HTTP <status> ..." when the server ANSWERED (a 400
        // from a cap or a 503 from the run gate is the service working as
        // designed), and anything else means the fetch itself failed.
        if (error.message.indexOf("HTTP ") === 0) {
          clearHeld();
          Render.refused(
            document.getElementById("view"),
            error.message,
            body,
            "refused"
          );
          // The service answered, so the form is still the way forward: put
          // it back under the refusal rather than making the operator find it.
          document.getElementById("view").appendChild(controls());
          setStatus(`The service refused the run: ${error.message}`, "failed");
        } else {
          noteTransportFailure(error.message, body);
          setStatus(
            `Nothing answered (${error.message}). ${fallbackNote()}`,
            "failed"
          );
        }
      })
      .then(function () {
        state.busy = false;
        button.disabled = false;
      });
  }

  /* -- Views -- */

  /** The views, in nav order. `needsRun` marks those that are about a run. */
  const VIEWS = [
    { id: "home", label: "Home", needsRun: false },
    { id: "session", label: "Session", needsRun: true },
    { id: "evidence", label: "Evidence", needsRun: true },
    { id: "proof", label: "Proof", needsRun: true },
    { id: "report", label: "Full report", needsRun: true },
    { id: "docs", label: "Documentation", needsRun: false },
  ];

  /** The view the URL asks for, defaulting to Home. */
  function routeId() {
    const asked = String(window.location.hash).replace("#/", "");
    if (asked === "run") {
      return "run";
    }
    let found = "home";
    VIEWS.forEach(function (entry) {
      if (entry.id === asked) {
        found = entry.id;
      }
    });
    return found;
  }

  /**
   * Navigate. Writing the hash triggers the render through `hashchange`, so
   * there is one path into a repaint.
   */
  function go(id) {
    if (String(window.location.hash) === `#/${id}`) {
      renderRoute();
      return;
    }
    window.location.hash = `#/${id}`;
  }

  /** Hold one run as the subject of every run-scoped view. */
  function hold(payload, meta) {
    state.payload = payload;
    state.payloadTitle = meta.title;
    state.payloadWhy = meta.why;
    state.payloadSource = meta.source;
    state.selectedFile = meta.file;
    state.autoplay = true;
  }

  /** Drop the held run, so nothing stale stands under a failure. */
  function clearHeld() {
    Render.stop();
    state.payload = null;
    state.selectedFile = null;
    state.autoplay = false;
    paintNav();
  }

  /**
   * Build the view tabs and mark the masthead's Run button. Run-scoped tabs are
   * disabled until a run is held.
   */
  function paintNav() {
    const here = routeId();
    const nav = document.getElementById("pages");
    if (nav) {
      nav.textContent = "";
      VIEWS.forEach(function (entry) {
        const current = entry.id === here;
        const button = h("button", {
          class: current ? "tab is-current" : "tab",
          text: entry.label,
          attrs: {
            type: "button",
            "aria-current": current ? "page" : "false",
          },
        });
        button.disabled = entry.needsRun && !state.payload;
        button.addEventListener("click", function () {
          go(entry.id);
        });
        nav.appendChild(button);
      });
    }
    const runLink = document.getElementById("run-link");
    if (runLink) {
      runLink.className = here === "run" ? "run-action is-current" : "run-action";
      runLink.setAttribute("aria-current", here === "run" ? "page" : "false");
    }
  }

  /** Whether the viewer asked the system for less motion. */
  function prefersStill() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  /** Render whatever the hash asks for. */
  function renderRoute() {
    const view = document.getElementById("view");
    const rail = document.getElementById("scenarios");
    const frame = document.getElementById("frame");
    if (!view) {
      return;
    }
    const here = routeId();
    const wide = here === "run" || here === "docs";
    paintNav();
    if (rail) {
      rail.textContent = "";
      rail.hidden = wide;
      if (!wide) {
        rail.appendChild(recordedList());
      }
    }
    if (frame) {
      frame.className = wide ? "frame is-wide" : "frame";
    }
    Render.stop();
    if (here === "run") {
      view.textContent = "";
      view.appendChild(controls());
      window.scrollTo(0, 0);
      return;
    }
    if (here === "home") {
      Render.home(view, homeReel(), context(), {
        autoplay: !prefersStill(),
        open: openReelItem,
      });
      window.scrollTo(0, 0);
      return;
    }
    if (here === "docs") {
      Render.docs(view, state.docs, context());
      window.scrollTo(0, 0);
      return;
    }
    if (!state.payload) {
      view.textContent = "";
      const start = h("button", {
        class: "run-button",
        text: "Run a session",
        attrs: { type: "button" },
      });
      start.addEventListener("click", function () {
        go("run");
      });
      view.appendChild(
        h("div", { class: "empty-state" }, [
          h("h1", { text: "Choose a scenario to replay" }),
          h("p", {
            text:
              "Pick one from the list, or run a new session against the " +
              "live service.",
          }),
          start,
        ])
      );
      return;
    }
    Render.page(view, here, state.payload, context(), {
      autoplay: here === "session" && state.autoplay,
      title: state.payloadTitle,
      description: state.payloadWhy,
      source: state.payloadSource,
    });
    if (here === "session") {
      state.autoplay = false;
    }
    window.scrollTo(0, 0);
  }

  /* -- Start-up -- */

  /** Show which mode the page is in, in the top bar, always. */
  function paintMode() {
    const chip = document.getElementById("mode-chip");
    if (!chip) {
      return;
    }
    chip.className = `mode-chip is-${state.mode}`;
    if (state.mode === "live") {
      chip.textContent = "LIVE API";
      chip.title =
        "The service answered, so a run started here is simulated on the " +
        "spot, not replayed from a recording.";
      return;
    }
    if (state.recordedHeld > 0) {
      chip.textContent = `RECORDED ONLY (${state.recordedHeld} held): API not reachable`;
      chip.title =
        "The service isn't answering. The recorded runs held in this " +
        "page's memory still render, and no live run can start.";
      return;
    }
    chip.className = "mode-chip is-dead";
    chip.textContent = "NOTHING LIVE: no API and no recorded runs";
    chip.title =
      "The service isn't answering and no recorded run loaded, so this " +
      "page has nothing to show. Start the server and reload.";
  }

  /**
   * How often the top bar re-checks that the service is really there, in ms.
   */
  const HEALTH_INTERVAL_MS = 5000;

  /**
   * Keep the top bar's claim true by ASKING, not by waiting to be surprised.
   */
  function watchService() {
    window.setInterval(function () {
      if (state.busy) {
        return;
      }
      getJson("/api/health")
        .then(function (health) {
          if (health && health.ok === true) {
            if (state.mode !== "live") {
              state.mode = "live";
              paintMode();
            }
            return;
          }
          if (state.mode !== "recorded") {
            state.mode = "recorded";
            paintMode();
          }
        })
        .catch(function () {
          if (state.mode !== "recorded") {
            state.mode = "recorded";
            paintMode();
          }
        });
    }, HEALTH_INTERVAL_MS);
  }

  /**
   * Wire the theme toggle. The theme goes on the ROOT element, where the colour
   * tokens are read, and changes colours and nothing else.
   */
  function wireTheme() {
    let theme = null;
    try {
      theme = window.localStorage.getItem(THEME_KEY);
    } catch (ignored) {
      theme = null;
    }
    if (theme !== "light" && theme !== "dark") {
      theme = window.matchMedia("(prefers-color-scheme: light)").matches
        ? "light"
        : "dark";
    }
    const button = document.getElementById("theme");

    function apply() {
      document.documentElement.setAttribute("data-theme", theme);
      if (!button) {
        return;
      }
      // The icon shows the theme a click switches TO.
      const next = theme === "dark" ? "light" : "dark";
      const label = `Switch to ${next} theme`;
      button.textContent = "";
      button.appendChild(
        Charts.icon(next === "light" ? "sun" : "moon", "theme-icon")
      );
      button.setAttribute("aria-label", label);
      button.title = label;
    }

    apply();
    if (!button) {
      return;
    }
    button.addEventListener("click", function () {
      theme = theme === "dark" ? "light" : "dark";
      apply();
      try {
        window.localStorage.setItem(THEME_KEY, theme);
      } catch (ignored) {
        /* Not remembered; the page still switched. */
      }
    });
  }

  /**
   * Load the contract, constants and documentation, then the API, falling back
   * to the recordings, then open on Home.
   */
  function boot() {
    wireTheme();
    const runLink = document.getElementById("run-link");
    if (runLink) {
      runLink.addEventListener("click", function () {
        go("run");
      });
    }
    // A printed report shows every section, not only the ones left open.
    window.addEventListener("beforeprint", function () {
      document.querySelectorAll("details").forEach(function (node) {
        node.open = true;
      });
    });
    Promise.all([
      getJson(`${STATIC}/data/api-contract.json`)
        .then(function (manifest) {
          Contract.install(manifest);
        })
        .catch(function () {
          /* The page still runs; the contract check passes everything. */
        }),
      getJson(`${STATIC}/data/constants.json`)
        .then(function (constants) {
          state.constants = constants;
        })
        .catch(function () {
          /* Views that needed one say the file was not served. */
        }),
      getJson(`${STATIC}/data/docs.json`)
        .then(function (docs) {
          state.docs = docs;
        })
        .catch(function () {
          /* Left null; the Documentation view says the file did not load. */
        }),
    ])
      .then(function () {
        return Promise.all([getJson("/api/attacks"), getJson("/api/defaults")])
          .then(function (both) {
            state.attacks = both[0];
            state.defaults = both[1];
            state.mode = "live";
          })
          .catch(function () {
            return Promise.all([
              getJson(`${STATIC}/data/recorded/attacks.json`),
              getJson(`${STATIC}/data/recorded/defaults.json`),
            ]).then(function (both) {
              state.attacks = both[0];
              state.defaults = both[1];
              state.mode = "recorded";
            });
          });
      })
      .then(function () {
        return getJson(`${STATIC}/data/recorded/index.json`)
          .then(function (index) {
            state.recorded = index;
          })
          .then(preloadRecorded);
      })
      .catch(function () {
        state.recorded = state.recorded || [];
      })
      .then(function () {
        paintMode();
        // Hold the first scenario that is actually in memory, so Session and
        // the other run views work from the first click.
        let opening = null;
        (state.recorded || []).forEach(function (entry) {
          if (opening === null && state.recordedPayloads[entry.file]) {
            opening = entry;
          }
        });
        if (opening !== null) {
          hold(state.recordedPayloads[opening.file], {
            title: opening.label,
            why: opening.why,
            source: "Recorded run",
            file: opening.file,
          });
        }
        window.addEventListener("hashchange", renderRoute);
        renderRoute();
        if (state.mode !== "live" && state.recordedHeld === 0) {
          // A cold load against a dead service. The page is here because the
          // browser had it cached; nothing behind it is.
          const view = document.getElementById("view");
          view.insertBefore(
            Render.banner("alarm", "⚠", "There is nothing to show", [
              "The service that serves this page isn't answering, so the " +
                "browser is probably showing a copy it cached earlier. " +
                "There's no API to run against and no recorded run loaded, " +
                "because the recordings come from that same process.",
              "Nothing on this page is a result. Start the server and reload.",
            ]),
            view.firstChild
          );
        }
        watchService();
      });
  }

  return { boot: boot, context: context, state: state };
})();

document.addEventListener("DOMContentLoaded", App.boot);
