/* format.js -- turning API numbers into strings, and nothing else. */

/* eslint-disable no-unused-vars */
const Fmt = (function () {
  "use strict";

  /** The mark used wherever a value is genuinely absent. */
  const ABSENT = "n/a";

  /** Return true when a value is a finite number the API actually supplied. */
  function present(value) {
    return typeof value === "number" && isFinite(value);
  }

  /** Render a probability or bound in scientific notation, four figures. */
  function exp(value) {
    if (!present(value)) {
      return ABSENT;
    }
    // JavaScript writes "1.0000e-9" where Python's {:.4e} writes "1.0000e-09".
    const parts = value.toExponential(4).split("e");
    const sign = parts[1].slice(0, 1);
    const digits = parts[1].slice(1);
    const pad = digits.length === 1 ? "0" : "";
    return `${parts[0]}e${sign}${pad}${digits}`;
  }

  /** Superscript glyphs for an exponent, so a bound reads as a number. */
  const SUPERSCRIPT = {
    0: "⁰",
    1: "¹",
    2: "²",
    3: "³",
    4: "⁴",
    5: "⁵",
    6: "⁶",
    7: "⁷",
    8: "⁸",
    9: "⁹",
    "-": "⁻",
    "+": "",
  };

  /** Render a probability for a room to read: `3.86 × 10⁻¹⁰`. */
  function sci(value) {
    if (!present(value)) {
      return ABSENT;
    }
    if (value === 0) {
      return "0";
    }
    const parts = value.toExponential(2).split("e");
    if (parts[1] === "+0") {
      return parts[0];
    }
    const power = parts[1]
      .split("")
      .map(function (char) {
        return SUPERSCRIPT[char];
      })
      .join("");
    return `${parts[0]} × 10${power}`;
  }

  /** Render a value with a fixed number of decimal places. */
  function fixed(value, places) {
    if (!present(value)) {
      return ABSENT;
    }
    return value.toFixed(places);
  }

  /** Render a rate (QBER, mismatch rate) as a rate. Never as a percentage. */
  function rate(value) {
    return fixed(value, 6);
  }

  /** Render a CHSH-scale quantity. */
  function chsh(value) {
    return fixed(value, 4);
  }

  /** Render an integer count with thousands separators. */
  function count(value) {
    if (!present(value)) {
      return ABSENT;
    }
    return value.toLocaleString("en-US");
  }

  /** Render a duration the API measured, in milliseconds. */
  function millis(value) {
    if (!present(value)) {
      return ABSENT;
    }
    return `${value.toFixed(1)} ms`;
  }

  /** Render an interval the API supplied, as an interval. */
  function interval(interval_, render) {
    if (!interval_) {
      return ABSENT;
    }
    const low = render(interval_.low);
    const high = render(interval_.high);
    const conf = interval_.confidence;
    const level = present(conf) ? conf.toFixed(2) : ABSENT;
    return `[${low}, ${high}]  ${interval_.method}, ${level}`;
  }

  /** Render a boolean the API supplied, with words rather than a tick. */
  function flag(value, yes, no) {
    if (value === true) {
      return yes;
    }
    if (value === false) {
      return no;
    }
    return ABSENT;
  }

  /** Render a list the API supplied, or say it is empty in words. */
  function list(items, emptyText) {
    if (!items || items.length === 0) {
      return emptyText;
    }
    return items.join(", ");
  }

  /** Render a `[party, message_bit]` link pair as a label. */
  function link(party, bit) {
    return `${party} / bit ${bit}`;
  }

  /** Render an API sentence for running prose. */
  function prose(text) {
    if (text === null || text === undefined) {
      return "";
    }
    return String(text).split(" -- ").join(", ");
  }

  return {
    ABSENT: ABSENT,
    chsh: chsh,
    count: count,
    exp: exp,
    fixed: fixed,
    flag: flag,
    interval: interval,
    link: link,
    list: list,
    millis: millis,
    present: present,
    prose: prose,
    rate: rate,
    sci: sci,
  };
})();
