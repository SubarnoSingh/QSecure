/* charts.js: inline SVG, hand-rolled, and the ONLY arithmetic in this app. */

const Charts = (function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";

  const LABEL_WIDTH = 148;
  const VALUE_WIDTH = 150;
  const RIGHT_PAD = 8;
  const TOP_PAD = 36;
  const AXIS_PAD = 30;
  const ROW_HEIGHT = 34;
  const BAR_HEIGHT = 13;
  const WIDTH = 760;

  // Every chart carries its own copy of the hatch pattern, so each needs its
  // own id.
  let patternSerial = 0;

  /** Create an SVG element with attributes. */
  function el(name, attrs) {
    const node = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  /** Create an SVG text node carrying a caller-supplied string. */
  function text(x, y, label, className, anchor) {
    const node = el("text", {
      x: x,
      y: y,
      class: className,
      "text-anchor": anchor || "start",
    });
    node.textContent = label;
    return node;
  }

  /** Keep a coordinate inside the plot rectangle. */
  function clamp(value, low, high) {
    if (value < low) {
      return low;
    }
    if (value > high) {
      return high;
    }
    return value;
  }

  /** Map a data value onto an x coordinate. The whole of the arithmetic. */
  function scale(value, domain, left, right) {
    const span = domain.max - domain.min;
    if (span === 0) {
      return left;
    }
    const fraction = (value - domain.min) / span;
    return clamp(left + fraction * (right - left), left, right);
  }

  /** Append the hatch pattern used for a cell that could not be evaluated. */
  function addHatch(svg, id) {
    const defs = el("defs", {});
    const pattern = el("pattern", {
      id: id,
      width: 8,
      height: 8,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    // Colours are classes, not attributes, so the theme lives in app.css.
    pattern.appendChild(
      el("rect", { class: "hatch-ground", width: 8, height: 8 })
    );
    pattern.appendChild(
      el("line", {
        class: "hatch-line",
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 8,
      })
    );
    defs.appendChild(pattern);
    svg.appendChild(defs);
  }

  /** Draw a horizontal bar chart with markers and whiskers. */
  function bars(options) {
    const rows = options.rows || [];
    const domain = options.domain;
    const ticks = options.ticks || [];
    const markers = options.markers || [];
    const left = LABEL_WIDTH;
    const right = WIDTH - VALUE_WIDTH - RIGHT_PAD;
    const plotHeight = ROW_HEIGHT * rows.length;
    const height = TOP_PAD + plotHeight + AXIS_PAD;
    patternSerial += 1;
    const hatchId = `hatch-withheld-${patternSerial}`;

    const svg = el("svg", {
      class: "chart",
      viewBox: `0 0 ${WIDTH} ${height}`,
      role: "img",
      "aria-label": options.title || "Chart",
    });
    const titleNode = el("title", {});
    titleNode.textContent = options.title || "Chart";
    svg.appendChild(titleNode);
    addHatch(svg, hatchId);

    ticks.forEach(function (tick) {
      const x = scale(tick.value, domain, left, right);
      svg.appendChild(
        el("line", {
          class: "grid-line",
          x1: x,
          y1: TOP_PAD,
          x2: x,
          y2: TOP_PAD + plotHeight,
        })
      );
      svg.appendChild(
        text(
          x,
          TOP_PAD + plotHeight + 16,
          tick.label,
          "tick-label",
          "middle"
        )
      );
    });

    svg.appendChild(
      el("line", {
        class: "axis-line",
        x1: left,
        y1: TOP_PAD,
        x2: left,
        y2: TOP_PAD + plotHeight,
      })
    );

    // Two label lanes, alternating, so that two markers close together on the
    // axis (the classical and Tsirelson bounds are 0.83 apart) do not print
    // their names on top of each other.
    markers.forEach(function (marker, index) {
      const x = scale(marker.value, domain, left, right);
      const lane = index % 2 === 0 ? 12 : 26;
      svg.appendChild(
        el("line", {
          class: marker.className,
          x1: x,
          y1: lane + 4,
          x2: x,
          y2: TOP_PAD + plotHeight,
        })
      );
      svg.appendChild(
        text(
          x,
          lane,
          marker.label,
          marker.labelClassName || "marker-label",
          "middle"
        )
      );
    });

    rows.forEach(function (row, index) {
      const top = TOP_PAD + ROW_HEIGHT * index;
      const mid = top + ROW_HEIGHT / 2;

      svg.appendChild(
        text(0, mid + 4, row.label, "row-label", "start")
      );

      if (row.unavailable) {
        svg.appendChild(
          el("rect", {
            class: "hatch-cell",
            x: left,
            y: top + 5,
            width: right - left,
            height: ROW_HEIGHT - 12,
            fill: `url(#${hatchId})`,
          })
        );
        svg.appendChild(
          text(
            left + (right - left) / 2,
            mid + 4,
            row.unavailable,
            "tick-label",
            "middle"
          )
        );
      } else if (row.value !== null && row.value !== undefined) {
        // Where a bar starts.
        const base =
          options.baseline === undefined ? domain.min : options.baseline;
        const zero = scale(base, domain, left, right);
        const x = scale(row.value, domain, left, right);
        // An SVG rect is anchored at its LEFT edge, so a bar running back from
        // the baseline starts at `x` and not at `zero`.
        svg.appendChild(
          el("rect", {
            class: row.className || "bar",
            x: Math.min(zero, x),
            y: mid - BAR_HEIGHT / 2,
            width: Math.abs(x - zero),
            height: BAR_HEIGHT,
          })
        );
        if (row.bound) {
          const lo = scale(row.bound.low, domain, left, right);
          const hi = scale(row.bound.high, domain, left, right);
          svg.appendChild(
            el("line", {
              class: "whisker-loose",
              x1: lo,
              y1: mid + BAR_HEIGHT,
              x2: hi,
              y2: mid + BAR_HEIGHT,
            })
          );
        }
        if (row.interval) {
          const lo = scale(row.interval.low, domain, left, right);
          const hi = scale(row.interval.high, domain, left, right);
          svg.appendChild(
            el("line", {
              class: "whisker",
              x1: lo,
              y1: mid - BAR_HEIGHT,
              x2: hi,
              y2: mid - BAR_HEIGHT,
            })
          );
          svg.appendChild(
            el("line", {
              class: "whisker",
              x1: lo,
              y1: mid - BAR_HEIGHT - 4,
              x2: lo,
              y2: mid - BAR_HEIGHT + 4,
            })
          );
          svg.appendChild(
            el("line", {
              class: "whisker",
              x1: hi,
              y1: mid - BAR_HEIGHT - 4,
              x2: hi,
              y2: mid - BAR_HEIGHT + 4,
            })
          );
        }
      }

      svg.appendChild(
        text(
          WIDTH - RIGHT_PAD,
          mid + 4,
          row.valueLabel,
          "value-label",
          "end"
        )
      );
    });

    return svg;
  }

  /* -- The bench -- */

  const XLINK = "http://www.w3.org/1999/xlink";
  const BENCH_WIDTH = 1000;
  const BENCH_HEIGHT = 520;

  /**
   * Join class names, skipping empty ones, so no class string carries a stray
   * space.
   */
  function classes(...names) {
    return names.filter(Boolean).join(" ");
  }

  /** Where the three parties stand. */
  const MOUNTS = {
    alice: { x: 20, y: 204, w: 264, h: 112 },
    bob: { x: 720, y: 24, w: 256, h: 132 },
    charlie: { x: 720, y: 364, w: 256, h: 132 },
  };

  /**
   * Every path a message can take, each drawn FROM its source TO its target.
   */
  const ROUTES = {
    "beam-bob": { d: "M 284 222 C 502 222, 502 62, 720 62", kind: "beam" },
    "cable-bob": { d: "M 284 262 C 502 262, 502 126, 720 126", kind: "cable" },
    "beam-charlie": {
      d: "M 284 298 C 502 298, 502 458, 720 458",
      kind: "beam",
    },
    forward: { d: "M 790 156 L 790 364", kind: "cable" },
    "private-down": { d: "M 910 156 L 910 364", kind: "private" },
    "private-up": { d: "M 910 364 L 910 156", kind: "private", hidden: true },
  };

  /** Where a spliced-in adversary sits on each route. */
  const SLOTS = {
    "beam-bob": [612, 79],
    "cable-bob": [612, 140],
    "beam-charlie": [612, 441],
    forward: [790, 236],
    "private-down": [910, 284],
  };

  /**
   * Append the bench's shared definitions: hazard stripes, glow, arrowhead, and
   * one addressable copy of every route for the particles to follow.
   */
  function benchDefs(svg, serial) {
    const defs = el("defs", {});

    const hazard = el("pattern", {
      id: `hazard-${serial}`,
      width: 12,
      height: 12,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    hazard.appendChild(
      el("rect", { class: "hazard-ground", width: 12, height: 12 })
    );
    hazard.appendChild(
      el("rect", { class: "hazard-stripe", width: 6, height: 12 })
    );
    defs.appendChild(hazard);

    const glow = el("filter", {
      id: `glow-${serial}`,
      x: "-100%",
      y: "-100%",
      width: "300%",
      height: "300%",
    });
    glow.appendChild(
      el("feGaussianBlur", { stdDeviation: 4, result: "blur" })
    );
    const merge = el("feMerge", {});
    merge.appendChild(el("feMergeNode", { in: "blur" }));
    merge.appendChild(el("feMergeNode", { in: "SourceGraphic" }));
    glow.appendChild(merge);
    defs.appendChild(glow);

    const arrow = el("marker", {
      id: `arrow-${serial}`,
      viewBox: "0 0 10 10",
      refX: 8,
      refY: 5,
      markerWidth: 6,
      markerHeight: 6,
      orient: "auto-start-reverse",
    });
    arrow.appendChild(
      el("path", { class: "arrow-head", d: "M 0 0 L 10 5 L 0 10 z" })
    );
    defs.appendChild(arrow);

    Object.keys(ROUTES).forEach(function (id) {
      defs.appendChild(
        el("path", { id: `route-${serial}-${id}`, d: ROUTES[id].d })
      );
    });

    svg.appendChild(defs);
  }

  /** Put moving particles on one live route. */
  function particles(svg, flow, serial) {
    const route = ROUTES[flow.route];
    const href = `#route-${serial}-${flow.route}`;
    const isBeam = route.kind === "beam";
    const count = isBeam ? 4 : 2;
    const seconds = isBeam ? 1.6 : 2.2;
    for (let index = 0; index < count; index += 1) {
      const node = isBeam
        ? el("circle", {
            class: "photon",
            r: 5,
            filter: `url(#glow-${serial})`,
          })
        : el("rect", {
            class: `pulse is-${route.kind}`,
            x: -10,
            y: -4,
            width: 20,
            height: 8,
            rx: 4,
          });
      const motion = el("animateMotion", {
        dur: `${seconds}s`,
        repeatCount: "indefinite",
        begin: `${(-seconds / count) * index}s`,
        rotate: "auto",
      });
      const mpath = el("mpath", { href: href });
      mpath.setAttributeNS(XLINK, "xlink:href", href);
      motion.appendChild(mpath);
      node.appendChild(motion);
      svg.appendChild(node);
    }
  }

  /**
   * Draw one party's mount: a rounded plate, an avatar at its left, the name
   * and role beside the avatar, and the state line underneath.
   */
  function mount(svg, party, serial) {
    const box = MOUNTS[party.id];
    if (!box) {
      return;
    }
    const signer = party.id === "alice" ? "is-signer" : "";
    svg.appendChild(
      el("rect", {
        class: classes(
          "mount",
          signer,
          party.className,
          party.scanning ? "is-scanning" : ""
        ),
        x: box.x,
        y: box.y,
        width: box.w,
        height: box.h,
        rx: 16,
      })
    );
    if (party.impersonated) {
      svg.appendChild(
        el("rect", {
          class: "mount-hazard",
          x: box.x,
          y: box.y,
          width: box.w,
          height: box.h,
          rx: 16,
          stroke: `url(#hazard-${serial})`,
        })
      );
    }
    // The avatar: a head and a pair of shoulders inside a circle.
    const cx = box.x + 44;
    const cy = box.y + 50;
    svg.appendChild(
      el("circle", {
        class: classes("avatar", signer, party.className),
        cx: cx,
        cy: cy,
        r: 22,
      })
    );
    svg.appendChild(
      el("circle", { class: "avatar-glyph", cx: cx, cy: cy - 6, r: 7 })
    );
    svg.appendChild(
      el("path", {
        class: "avatar-glyph",
        d: `M ${cx - 12} ${cy + 15} C ${cx - 12} ${cy + 4}, ${cx + 12} ${
          cy + 4
        }, ${cx + 12} ${cy + 15}`,
      })
    );
    svg.appendChild(
      text(box.x + 80, box.y + 48, party.name, "mount-name", "start")
    );
    svg.appendChild(
      text(box.x + 80, box.y + 74, party.role || "", "mount-role", "start")
    );
    if (party.stateLabel) {
      svg.appendChild(
        text(
          box.x + 24,
          box.y + box.h - 22,
          party.stateLabel,
          classes("mount-state", party.className),
          "start"
        )
      );
    }
  }

  /**
   * Draw one adversary as a component spliced into the route it holds: a
   * hazard-striped plate, its label, and a warning triangle left of the label.
   */
  function component(svg, adversary, serial) {
    const slot = SLOTS[adversary.route];
    if (!slot) {
      return;
    }
    const x = slot[0];
    const y = slot[1];
    svg.appendChild(
      el("rect", {
        class: classes("adversary", adversary.className),
        x: x - 84,
        y: y - 20,
        width: 168,
        height: 40,
        rx: 10,
        stroke: `url(#hazard-${serial})`,
      })
    );
    // The label stays the plate's next sibling, so a rule written as
    // `.adversary.is-idle + .adversary-label` still reaches it.
    svg.appendChild(
      text(x + 14, y + 5, adversary.label || "", "adversary-label", "middle")
    );
    svg.appendChild(
      el("path", {
        class: classes("adversary-glyph", adversary.className),
        d: `M ${x - 66} ${y - 10} L ${x - 55} ${y + 9} L ${x - 77} ${
          y + 9
        } Z M ${x - 66} ${y - 3} L ${x - 66} ${y + 3}`,
      })
    );
  }

  /** Draw the session on the bench. */
  function bench(options) {
    patternSerial += 1;
    const serial = patternSerial;
    const glow = `url(#glow-${serial})`;
    const noisy = {};
    (options.noisy || []).forEach(function (id) {
      noisy[id] = true;
    });

    const svg = el("svg", {
      class: "bench-svg",
      viewBox: `0 0 ${BENCH_WIDTH} ${BENCH_HEIGHT}`,
      role: "img",
      "aria-label": options.title || "The session",
    });
    const titleNode = el("title", {});
    titleNode.textContent = options.title || "The session";
    svg.appendChild(titleNode);
    benchDefs(svg, serial);

    Object.keys(ROUTES).forEach(function (id) {
      const route = ROUTES[id];
      if (route.hidden) {
        return;
      }
      const attrs = {
        class: classes("route", `is-${route.kind}`, noisy[id] ? "is-noisy" : ""),
        d: route.d,
      };
      if (route.kind === "beam") {
        attrs.filter = glow;
      }
      svg.appendChild(el("path", attrs));
    });

    (options.live || []).forEach(function (flow) {
      const route = ROUTES[flow.route];
      if (!route) {
        return;
      }
      const attrs = {
        class: `route-live is-${route.kind}`,
        d: route.d,
        "marker-end": `url(#arrow-${serial})`,
      };
      if (route.kind === "beam") {
        attrs.filter = glow;
      }
      svg.appendChild(el("path", attrs));
    });

    (options.live || []).forEach(function (flow) {
      if (ROUTES[flow.route]) {
        particles(svg, flow, serial);
      }
    });

    (options.parties || []).forEach(function (party) {
      mount(svg, party, serial);
    });

    (options.adversaries || []).forEach(function (adversary) {
      component(svg, adversary, serial);
    });

    return svg;
  }

  /* -- Icons -- */

  /** Line icons on a 24 px grid, as `[class, d]` pairs. */
  const ICONS = {
    shield: [
      [
        "icon-stroke",
        "M 12 3 L 19.5 5.75 V 11.25 C 19.5 15.75 16.4 19.5 12 21 C 7.6 19.5 4.5 15.75 4.5 11.25 V 5.75 Z",
      ],
      ["icon-stroke icon-detail", "M 8.5 12 L 11 14.5 L 15.5 9.5"],
    ],
    seal: [
      ["icon-stroke", "M 6 9.5 A 6 6 0 1 0 18 9.5 A 6 6 0 1 0 6 9.5 Z"],
      ["icon-stroke", "M 8.5 14.4 L 7 21 L 12 18.5 L 17 21 L 15.5 14.4"],
      ["icon-stroke icon-detail", "M 9.5 9.5 L 11.25 11.25 L 14.5 8"],
    ],
    handoff: [
      ["icon-stroke", "M 2.5 19.5 A 2 2 0 1 0 6.5 19.5 A 2 2 0 1 0 2.5 19.5 Z"],
      [
        "icon-stroke",
        "M 17.5 19.5 A 2 2 0 1 0 21.5 19.5 A 2 2 0 1 0 17.5 19.5 Z",
      ],
      ["icon-stroke", "M 5 15.5 C 8 10.5, 16 10.5, 19 15.5"],
      ["icon-stroke", "M 15.8 14.6 L 19 15.5 L 20.1 12.3"],
      ["icon-stroke icon-detail", "M 9 2 H 13 L 15 4 V 9 H 9 Z"],
    ],
    sun: [
      ["icon-stroke", "M 8 12 A 4 4 0 1 0 16 12 A 4 4 0 1 0 8 12 Z"],
      [
        "icon-stroke icon-detail",
        "M 12 2.5 V 4.5 M 12 19.5 V 21.5 M 2.5 12 H 4.5 M 19.5 12 H 21.5 M 5.3 5.3 L 6.7 6.7 M 17.3 17.3 L 18.7 18.7 M 5.3 18.7 L 6.7 17.3 M 17.3 6.7 L 18.7 5.3",
      ],
    ],
    moon: [
      ["icon-stroke", "M 20.5 14 A 8.5 8.5 0 1 1 10 3.5 A 7.6 7.6 0 0 0 20.5 14 Z"],
    ],
    download: [
      ["icon-stroke", "M 4 15 V 19.5 H 20 V 15"],
      ["icon-stroke icon-detail", "M 12 3.5 V 14.5 M 7.5 10 L 12 14.5 L 16.5 10"],
    ],
    print: [
      ["icon-stroke", "M 7 17.5 H 4 V 9.5 H 20 V 17.5 H 17"],
      ["icon-stroke", "M 7 9.5 V 3.5 H 17 V 9.5"],
      ["icon-stroke icon-detail", "M 7 13.5 H 17 V 21 H 7 Z"],
    ],
    play: [["icon-stroke", "M 8 4.5 L 19 12 L 8 19.5 Z"]],
    alert: [
      ["icon-stroke", "M 12 3.5 L 21.5 20 H 2.5 Z"],
      ["icon-stroke icon-detail", "M 12 9.5 V 14 M 12 16.75 V 17.25"],
    ],
  };

  /**
   * Draw a named line icon. Decorative: hidden from assistive technology, so
   * whatever carries it must say the same thing in words.
   */
  function icon(name, className) {
    const svg = el("svg", {
      class: classes("icon", className),
      viewBox: "0 0 24 24",
      "aria-hidden": "true",
      focusable: "false",
    });
    const parts = Object.prototype.hasOwnProperty.call(ICONS, name)
      ? ICONS[name]
      : [];
    parts.forEach(function (part) {
      svg.appendChild(el("path", { class: part[0], d: part[1] }));
    });
    return svg;
  }

  return { bars: bars, bench: bench, icon: icon };
})();
