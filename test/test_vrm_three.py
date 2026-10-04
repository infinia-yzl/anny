# OpenSculptBoy
# Apache License, Version 2.0
"""
The shared VRM exports (test/vrm_fixtures.py) in three-vrm and in the Khronos glTF validator.

three-vrm is the VRM loader of three.js that web VTuber apps build on. The page
viewer/test/vrm_page/main.ts loads each export with it in Chromium (SwiftShader, through
Playwright), and the tests check what three-vrm makes of the file:

- the meta version, every humanoid bone of the file (raw and normalised) and the expressions: the
  presets of the expression table and the 52 perfect-sync customs in PascalCase;
- a frame renders the figure without console errors;
- bone look-at: a yaw of 20 degrees turns the eye bones by the range maps of the file, the
  eyeballs turn with their bones, and ``EyeLookOutLeft`` at 1 then moves the lids but not the
  eyeballs, so that the eyes do not turn twice (the ``eyeLook*`` targets of a VRM file are
  lid-only);
- the twist: a roll of 90 degrees of the normalised ``leftHand`` turns the forearm twist node
  ``lowerarm02.L`` by about 45 degrees through its VRM 1.0 roll constraint, and the hand by 90;
  VRM 0.x has no constraint (its twist weights move to the mapped bones), and the twist node
  stays still.

The Khronos glTF validator (the gltf-validator package) must report no error in either file; the
test prints its warnings.

The page is bundled at test time with the viewer's esbuild, so three-vrm never enters the viewer
page. The checks need the viewer's node packages (``cd viewer && npm ci``), Playwright and
Chromium; each check skips with its reason when one is missing.
"""

import functools
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from opensculptboy.export import vrm_tables
from test import vrm_fixtures
from test.gltf_reader import GLB
from test.markers import local_only

REPO = pathlib.Path(__file__).resolve().parents[1]
VIEWER = REPO / "viewer"
PAGE = VIEWER / "test" / "vrm_page"
NODE_MODULES = VIEWER / "node_modules"
CHROMIUM = pathlib.Path("/opt/pw-browsers/chromium")
# The page is served from this made-up origin through Playwright's request routing.
ORIGIN = "http://vrm-page.test/"
CONTENT_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".vrm": "application/octet-stream",
}
YAW = 20.0  # degrees
# VRM 0.x preset names -> the VRM 1.0 names that three-vrm gives them.
V0_TO_V1 = {
    "a": "aa",
    "i": "ih",
    "u": "ou",
    "e": "ee",
    "o": "oh",
    "blink": "blink",
    "blink_l": "blinkLeft",
    "blink_r": "blinkRight",
    "joy": "happy",
    "angry": "angry",
    "sorrow": "sad",
    "fun": "relaxed",
    "lookup": "lookUp",
    "lookdown": "lookDown",
    "lookleft": "lookLeft",
    "lookright": "lookRight",
    "neutral": "neutral",
}
# The validator's severities.
ERROR, WARNING, INFO, HINT = 0, 1, 2, 3


def missing_node_package(package: str) -> str | None:
    """Why node cannot run ``package`` from the viewer's node packages, or None."""
    if shutil.which("node") is None:
        return "node is not installed"
    if not (NODE_MODULES / package / "package.json").exists():
        return f"the viewer's node packages lack {package} (cd viewer && npm ci)"
    return None


def missing_page_tools() -> str | None:
    """Why the three-vrm page cannot run, or None."""
    reason = missing_node_package("@pixiv/three-vrm") or missing_node_package("esbuild")
    if reason:
        return reason
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return "Playwright is not installed"
    return None


_PAGE_DIR = tempfile.TemporaryDirectory()


@functools.lru_cache(maxsize=1)
def page_dir() -> pathlib.Path:
    """A folder with the page: index.html and main.ts bundled as vrm_page.js."""
    out = pathlib.Path(_PAGE_DIR.name)
    esbuild = NODE_MODULES / ".bin" / "esbuild"
    subprocess.run(
        [
            str(esbuild),
            str(PAGE / "main.ts"),
            "--bundle",
            "--format=esm",
            "--target=es2022",
            f"--outfile={out / 'vrm_page.js'}",
            "--log-level=warning",
        ],
        cwd=VIEWER,
        check=True,
        capture_output=True,
    )
    shutil.copy(PAGE / "index.html", out / "index.html")
    return out


def rotation_angle(q_from, q_to) -> tuple[float, np.ndarray]:
    """The angle (degrees) and the world axis of the rotation from ``q_from`` to ``q_to``."""
    a, b = np.asarray(q_from, dtype=np.float64), np.asarray(q_to, dtype=np.float64)
    # q_to * conj(q_from), quaternions as [x, y, z, w]
    x1, y1, z1, w1 = b
    x2, y2, z2, w2 = -a[0], -a[1], -a[2], a[3]
    q = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    angle = float(np.degrees(2 * np.arctan2(s, q[3])))
    return angle, (q[:3] / s if s > 1e-12 else np.zeros(3))


def kabsch_angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (degrees) of the rotation that best maps the points ``a`` to ``b``."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def humanoid_nodes(doc: dict) -> dict[str, int]:
    """VRM 1.0 bone name -> node index, for a VRM 1.0 or 0.x glTF JSON."""
    if "VRMC_vrm" in doc.get("extensions", {}):
        bones = doc["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
        return {name: entry["node"] for name, entry in bones.items()}
    rename = {"ThumbProximal": "ThumbMetacarpal", "ThumbIntermediate": "ThumbProximal"}
    result = {}
    for entry in doc["extensions"]["VRM"]["humanoid"]["humanBones"]:
        name = entry["bone"]
        for side in ("left", "right"):
            if name.startswith(side) and name[len(side) :] in rename:
                name = side + rename[name[len(side) :]]
        result[name] = entry["node"]
    return result


def eye_ranges(doc: dict) -> dict[str, float]:
    """The eye-bone angle (degrees) that a yaw of ``YAW`` gives: "outer" and "inner"."""
    if "VRMC_vrm" in doc.get("extensions", {}):
        look = doc["extensions"]["VRMC_vrm"]["lookAt"]
        maps = {
            "outer": look["rangeMapHorizontalOuter"],
            "inner": look["rangeMapHorizontalInner"],
        }
        return {
            k: m["outputScale"] * min(YAW / m["inputMaxValue"], 1.0)
            for k, m in maps.items()
        }
    first = doc["extensions"]["VRM"]["firstPerson"]
    maps = {
        "outer": first["lookAtHorizontalOuter"],
        "inner": first["lookAtHorizontalInner"],
    }
    # three-vrm reads a 0.x degree map as linear: yRange at xRange
    return {k: m["yRange"] * min(YAW / m["xRange"], 1.0) for k, m in maps.items()}


class _ThreeVrm:
    """The checks of one VRM version in three-vrm; ``VERSION`` names the version."""

    VERSION = "1.0"

    @classmethod
    def setUpClass(cls):
        reason = missing_page_tools()
        if reason:
            raise unittest.SkipTest(reason)
        from playwright.sync_api import sync_playwright

        path, _ = vrm_fixtures.vrm_export(cls.VERSION)
        cls.doc = vrm_fixtures.vrm_json(cls.VERSION)
        cls.prepare(GLB(path))
        out = page_dir()
        shutil.copy(path, out / f"avatar_{cls.VERSION}.vrm")
        cls.console, cls.page_errors = [], []
        args = [
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
        ]
        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            try:
                browser = p.chromium.launch(args=args, **kw)
            except Exception as error:  # no browser for Playwright
                raise unittest.SkipTest(f"Chromium does not start: {error}")
            try:
                tab = browser.new_page(viewport=dict(width=320, height=320))

                def serve(route):
                    name = route.request.url[len(ORIGIN) :].split("?")[0]
                    file = out / name
                    if file.is_file():
                        route.fulfill(
                            path=str(file),
                            content_type=CONTENT_TYPES.get(
                                file.suffix, "application/octet-stream"
                            ),
                        )
                    else:
                        route.fulfill(status=404, body="")

                tab.route(ORIGIN + "**", serve)
                tab.on("console", lambda m: cls.console.append((m.type, m.text)))
                tab.on("pageerror", lambda e: cls.page_errors.append(str(e)))
                tab.goto(f"{ORIGIN}index.html?vrm=avatar_{cls.VERSION}.vrm")
                tab.wait_for_function("window.__VRM_READY === true", timeout=600_000)
                error = tab.evaluate("() => window.__VRM_ERROR || null")
                if error:
                    raise AssertionError(f"three-vrm could not load the file: {error}")
                cls.info = tab.evaluate("() => window.__VRM_INFO")
                cls.frame = tab.evaluate("() => window.vrmRender()")
                cls.experiments(tab)
            finally:
                browser.close()

    @classmethod
    def prepare(cls, glb: GLB):
        """The vertex and node indices that the page checks use."""
        g = glb.json
        human = humanoid_nodes(g)
        # the primitive that draws the eyeballs: the vertices that follow an eye bone alone
        cls.eyes = None
        for node in g["nodes"]:
            if "mesh" not in node or "skin" not in node:
                continue
            joints = g["skins"][node["skin"]]["joints"]
            if not all(human.get(f"{s}Eye", -1) in joints for s in ("left", "right")):
                continue
            for p, prim in enumerate(g["meshes"][node["mesh"]]["primitives"]):
                j = glb.accessor(prim["attributes"]["JOINTS_0"]).astype(int)
                w = glb.accessor(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
                eyes = {}
                for side in ("left", "right"):
                    slot = joints.index(human[f"{side}Eye"])
                    eyes[side] = np.flatnonzero(((j == slot) * w).sum(axis=1) > 0.999)
                if len(eyes["left"]) and len(eyes["right"]):
                    names = (
                        g["meshes"][node["mesh"]]
                        .get("extras", {})
                        .get("targetNames", [])
                    )
                    lids = np.array([], dtype=int)
                    if "eyeLookOutLeft" in names:
                        target = prim["targets"][names.index("eyeLookOutLeft")]
                        delta = np.linalg.norm(glb.accessor(target["POSITION"]), axis=1)
                        moving = np.setdiff1d(
                            np.flatnonzero(delta > 1e-5), eyes["left"]
                        )
                        lids = moving[np.argsort(-delta[moving])][:64]
                    cls.eyes = dict(mesh=node["mesh"], primitive=p, lids=lids, **eyes)
                    break
            if cls.eyes:
                break
        names = [n.get("name") for n in g["nodes"]]
        cls.twist_nodes = [
            names.index(n) if n in names else -1
            for n in ("lowerarm02.L", "wrist.L", "lowerarm01.L")
        ]

    @classmethod
    def experiments(cls, tab):
        """The look-at, expression and roll experiments; the tests check the results."""

        def vertices(indices):
            e = cls.eyes
            result = tab.evaluate(
                "([m, p, i]) => window.vrmVertices(m, p, i)",
                [e["mesh"], e["primitive"], [int(i) for i in indices]],
            )
            return None if result is None else np.array(result, dtype=np.float64)

        cls.look = None
        if cls.eyes is not None and cls.info["lookAt"] is not None:
            e = cls.eyes
            watched = np.concatenate([e["left"], e["right"], e["lids"]])
            look = dict(rest=tab.evaluate("() => window.vrmLookAt(0, 0)"))
            look["rest_vertices"] = vertices(watched)
            look["turned"] = tab.evaluate(f"() => window.vrmLookAt({YAW}, 0)")
            look["turned_vertices"] = vertices(watched)
            look["weight"] = tab.evaluate(
                "() => window.vrmExpression('EyeLookOutLeft', 1)"
            )
            look["both_vertices"] = vertices(watched)
            cls.look = look
        # a roll of the hand; the frame is drawn with the hand rolled and the expression on
        cls.roll = None
        call = "([a, n]) => window.vrmRoll('leftHand', [1, 0, 0], a, n)"
        if min(cls.twist_nodes) >= 0:
            cls.roll = dict(
                rest=tab.evaluate(call, [0, cls.twist_nodes]),
                rolled=tab.evaluate(call, [90, cls.twist_nodes]),
            )
        cls.posed_frame = tab.evaluate("() => window.vrmRender()")

    # --- the checks --------------------------------------------------------------------------

    def test_meta_and_humanoid(self):
        self.assertEqual(
            self.info["metaVersion"], {"1.0": "1", "0.x": "0"}[self.VERSION]
        )
        self.assertEqual(self.info["name"], vrm_fixtures.CHARACTER.name)
        expected = set(vrm_tables.humanoid_bones("1.0"))
        self.assertEqual(set(self.info["humanBones"]), expected)
        self.assertEqual(set(self.info["normalizedBones"]), expected)
        self.assertLessEqual(
            vrm_tables.required_bones("1.0"), set(self.info["humanBones"])
        )

    def file_expressions(self) -> tuple[set[str], set[str]]:
        """The preset and custom expression names of the file, as three-vrm names them."""
        if self.VERSION == "1.0":
            e = self.doc["extensions"]["VRMC_vrm"].get("expressions", {})
            return set(e.get("preset", {})), set(e.get("custom", {}))
        groups = self.doc["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
        presets = {
            V0_TO_V1[g["presetName"]] for g in groups if g["presetName"] != "unknown"
        }
        customs = {g["name"] for g in groups if g["presetName"] == "unknown"}
        return presets, customs

    def table_expressions(self, labels) -> tuple[set[str], set[str]]:
        """The preset and custom names of the expression table, as three-vrm names them."""
        table = vrm_tables.expressions(self.VERSION, labels)
        if self.VERSION == "1.0":
            return (
                {e.name for e in table if e.preset},
                {e.name for e in table if not e.preset},
            )
        # VRM 0.x: a preset without a 0.x preset (surprised) becomes a custom group
        return (
            {V0_TO_V1[e.vrm0_preset] for e in table if e.vrm0_preset},
            {e.vrm0_name or e.name for e in table if not e.vrm0_preset},
        )

    def test_expressions(self):
        presets, customs = self.file_expressions()
        self.assertEqual(set(self.info["presets"]), presets)
        self.assertEqual(set(self.info["customs"]), customs)
        labels = self.doc["meshes"][0]["extras"]["targetNames"]
        self.assertEqual((presets, customs), self.table_expressions(labels))
        perfect_sync = {label[0].upper() + label[1:] for label in labels}
        self.assertEqual(len(perfect_sync), 52)
        self.assertLessEqual(perfect_sync, customs, "the perfect-sync customs")

    def test_frame_renders_without_errors(self):
        errors = [text for kind, text in self.console if kind == "error"]
        self.assertEqual(errors + self.page_errors, [])
        for frame in (self.frame, self.posed_frame):
            self.assertEqual(frame["glError"], 0)
            self.assertGreaterEqual(frame["calls"], 1)
            self.assertGreater(frame["covered"], 0.02, "the figure covers the frame")
        self.assertGreaterEqual(
            self.info["mtoonMaterials"], 1, "three-vrm made no MToon"
        )

    def test_look_at_turns_the_eye_bones(self):
        self.assertIsNotNone(self.info["lookAt"], "three-vrm found no look-at")
        self.assertEqual(self.info["lookAt"]["type"], "bone")
        self.assertIsNotNone(self.eyes, "no vertices follow the eye bones alone")
        look, expected = self.look, eye_ranges(self.doc)
        for side, key in (("leftEye", "outer"), ("rightEye", "inner")):
            angle, _ = rotation_angle(look["rest"][side], look["turned"][side])
            self.assertAlmostEqual(angle, expected[key], delta=0.05, msg=side)
        # the eyeballs turn with their bones
        n = len(self.eyes["left"])
        rest, turned = look["rest_vertices"], look["turned_vertices"]
        angle = kabsch_angle(rest[:n], turned[:n])
        self.assertAlmostEqual(angle, expected["outer"], delta=0.2)

    def test_eye_look_expression_does_not_turn_the_eyeballs_twice(self):
        self.assertIsNotNone(self.look, "three-vrm found no look-at or no eyeballs")
        look = self.look
        self.assertEqual(look["weight"], 1.0)
        eyes = len(self.eyes["left"]) + len(self.eyes["right"])
        turned, both = look["turned_vertices"], look["both_vertices"]
        moved = np.linalg.norm(both - turned, axis=1)
        self.assertLess(moved[:eyes].max(), 2e-5, "EyeLookOutLeft moves the eyeballs")
        self.assertTrue(len(self.eyes["lids"]), "EyeLookOutLeft moves no lid vertex")
        self.assertGreater(moved[eyes:].max(), 1e-4, "EyeLookOutLeft moves no lid")


@local_only("Playwright, Chromium and the viewer's node packages (three-vrm)")
class TestThreeVrm1(_ThreeVrm, unittest.TestCase):
    VERSION = "1.0"

    def test_hand_roll_twists_the_forearm(self):
        constrained = {
            n["name"]
            for n in self.doc["nodes"]
            if "VRMC_node_constraint" in n.get("extensions", {})
        }
        if "lowerarm02.L" not in constrained:
            self.skipTest(
                "the VRM 1.0 export carries no roll constraint on lowerarm02.L"
            )
        self.assertGreaterEqual(self.info["constraints"], len(constrained))
        self.assertIsNotNone(self.roll)
        rest, rolled = self.roll["rest"], self.roll["rolled"]
        twist, twist_axis = rotation_angle(rest[0], rolled[0])
        hand, _ = rotation_angle(rest[1], rolled[1])
        forearm, _ = rotation_angle(rest[2], rolled[2])
        self.assertAlmostEqual(hand, 90.0, delta=0.5)
        self.assertAlmostEqual(twist, 45.0, delta=2.0)
        self.assertGreater(
            abs(twist_axis[0]), 0.99, "the twist turns about the forearm"
        )
        self.assertLess(forearm, 0.5)


@local_only("Playwright, Chromium and the viewer's node packages (three-vrm)")
class TestThreeVrm0(_ThreeVrm, unittest.TestCase):
    VERSION = "0.x"

    def test_hand_roll_without_constraints(self):
        """
        VRM 0.x has no node constraints (its twist weights move to the mapped bones): a roll of
        the hand turns the hand alone.
        """
        self.assertEqual(self.info["constraints"], 0)
        self.assertIsNotNone(self.roll)
        rest, rolled = self.roll["rest"], self.roll["rolled"]
        twist, _ = rotation_angle(rest[0], rolled[0])
        hand, _ = rotation_angle(rest[1], rolled[1])
        self.assertAlmostEqual(hand, 90.0, delta=0.5)
        self.assertLess(twist, 0.5)


def validate(paths: list[pathlib.Path]) -> list[dict]:
    """The reports of the Khronos glTF validator (viewer/test/vrm_page/validate.mjs)."""
    result = subprocess.run(
        ["node", str(PAGE / "validate.mjs"), *map(str, paths)],
        cwd=VIEWER,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@local_only("the viewer's node packages (gltf-validator)")
class TestKhronosValidator(unittest.TestCase):
    """The glTF core of both files passes the Khronos validator without an error."""

    def setUp(self):
        reason = missing_node_package("gltf-validator")
        if reason:
            self.skipTest(reason)

    def check(self, version: str):
        path, _ = vrm_fixtures.vrm_export(version)
        (report,) = validate([path])
        by_code: dict[tuple[int, str], int] = {}
        for m in report["messages"]:
            by_code[m["severity"], m["code"]] = (
                by_code.get((m["severity"], m["code"]), 0) + 1
            )
        names = {ERROR: "error", WARNING: "warning", INFO: "info", HINT: "hint"}
        print(
            f"\nKhronos glTF validator {report['validatorVersion']}, VRM {version}: "
            f"{report['numErrors']} errors, {report['numWarnings']} warnings, "
            f"{report['numInfos']} infos, {report['numHints']} hints"
        )
        for (severity, code), count in sorted(by_code.items()):
            print(f"  {names[severity]} {code} x{count}")
        errors = [m for m in report["messages"] if m["severity"] == ERROR]
        self.assertEqual(errors[:20], [], f"{report['numErrors']} validator errors")

    def test_vrm1(self):
        self.check("1.0")

    def test_vrm0(self):
        self.check("0.x")


if __name__ == "__main__":
    unittest.main()
