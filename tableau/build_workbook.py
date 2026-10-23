"""Generate the Tableau workbook: tableau/NJ_Mortgage_Lending.twb plus Hyper extracts in tableau/extracts/.

Run after `python -m mortgage.run export`:
    python tableau/build_workbook.py
Then open the .twb in Tableau Public and use File > Save to Tableau Public As...

Design: a dark header band, a row of KPI tiles ("big numbers"), white chart cards on a light
canvas, Tableau's own font family, and interactive filters/highlights. Every data source is a
Hyper extract (Tableau Public only publishes extracts). Headlines, KPI tiles and story captions
are computed from reports/metrics/*.json and the exports, so they always match the analysis.

Set TABLEAU_ACTIVE=<dashboard name> to open on a specific dashboard (used for visual checks).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "exports" / "tableau"
EXTRACT_DIR = ROOT / "tableau" / "extracts"
DERIVED_DIR = EXTRACT_DIR / "csv"
METRICS_DIR = ROOT / "reports" / "metrics"
OUT = ROOT / "tableau" / "NJ_Mortgage_Lending.twb"
YEAR = 2024
STORY = "New Jersey Mortgage Lending"

# Theme. Categorical colors come from Tableau's default palette (blue, orange, red, teal...), assigned
# in alphabetical order of values, so category names are chosen to land the highlight on orange.
NAVY, CANVAS, CARD = "#14213d", "#f2f3f5", "#ffffff"
TAB_BLUE, TAB_ORANGE = "#4e79a7", "#f28e2b"
INK, INK_2, MUTED, ON_DARK, ON_DARK_2 = "#1b1b1b", "#555555", "#8a8a8a", "#ffffff", "#c9d1e3"
H_FONT, B_FONT, L_FONT = "Tableau Semibold", "Tableau Book", "Tableau Light"
W, H = 1300, 860   # dashboard size in pixels


def a(v) -> str:
    return quoteattr(str(v))


# ================================================================== extracts
def infer_types(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:5001]
    types = {}
    for i, col in enumerate(header):
        vals = [r[i] for r in body if i < len(r) and r[i] != ""]

        def all_match(fn) -> bool:
            try:
                return bool(vals) and all(fn(v) for v in vals)
            except ValueError:
                return False

        types[col] = ("integer" if all_match(lambda v: str(int(v)) == v)
                      else "real" if all_match(lambda v: float(v) == float(v)) else "string")
    return types


def build_extract(path: Path, columns: dict[str, str]) -> Path:
    """CSV -> Hyper extract with typed columns in "Extract"."Extract" (the layout Tableau expects)."""
    from tableauhyperapi import (Connection, CreateMode, HyperProcess, Inserter, SqlType, TableDefinition,
                                 TableName, Telemetry)

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXTRACT_DIR / path.name.replace(".csv", ".hyper")
    sql = {"integer": SqlType.big_int(), "real": SqlType.double(), "string": SqlType.text()}
    cast = {"integer": lambda v: int(float(v)), "real": float, "string": str}
    table = TableDefinition(TableName("Extract", "Extract"), [TableDefinition.Column(c, sql[t]) for c, t in columns.items()])
    with path.open(encoding="utf-8") as f:
        rows = [[None if r[c] == "" else cast[t](r[c]) for c, t in columns.items()] for r in csv.DictReader(f)]
    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hp:
        with Connection(hp.endpoint, target, CreateMode.CREATE_AND_REPLACE) as con:
            con.catalog.create_schema("Extract")
            con.catalog.create_table(table)
            with Inserter(con, table) as ins:
                ins.add_rows(rows)
                ins.execute()
    return target


def write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = DERIVED_DIR / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ================================================================== data sources
@dataclass
class Calc:
    caption: str
    datatype: str
    role: str  # "dimension" | "measure"
    formula: str


@dataclass
class DataSource:
    key: str
    path: Path
    caption: str
    dims: set[str] = dc_field(default_factory=set)
    calcs: dict[str, Calc] = dc_field(default_factory=dict)
    semantic: dict[str, str] = dc_field(default_factory=dict)
    captions: dict[str, str] = dc_field(default_factory=dict)
    sequential: dict = dc_field(default_factory=dict)   # instance name -> Tableau palette name
    formats: dict = dc_field(default_factory=dict)      # column -> default number format

    def __post_init__(self):
        self.name = f"federated.{self.key}"
        self.columns = infer_types(self.path)

    def meta(self, col: str) -> tuple[str, str, str, str]:
        """(caption, datatype, role, type)"""
        if col in self.calcs:
            c = self.calcs[col]
            return c.caption, c.datatype, c.role, "quantitative" if c.role == "measure" else "nominal"
        dt = self.columns[col]
        caption = self.captions.get(col, col.replace("_", " ").capitalize())
        if dt == "string" or col in self.dims:
            return caption, dt, "dimension", "nominal" if dt == "string" else "ordinal"
        return caption, dt, "measure", "quantitative"

    def column_xml(self, col: str, indent: str) -> str:
        caption, dt, role, typ = self.meta(col)
        sem = f" semantic-role={a(self.semantic[col])}" if col in self.semantic else ""
        fmt = f" default-format={a(self.formats[col])}" if col in self.formats else ""
        head = f"{indent}<column caption={a(caption)} datatype={a(dt)}{fmt} name={a('[' + col + ']')} role={a(role)} type={a(typ)}{sem}"
        if col in self.calcs:
            return f"{head}>\n{indent}  <calculation class='tableau' formula={a(self.calcs[col].formula)} />\n{indent}</column>"
        return head + " />"

    def style_xml(self) -> str:
        if not self.sequential:
            return ""
        rules = [f"        <encoding attr='color' field={a(inst)} palette={a(p)} type='interpolated' />" for inst, p in self.sequential.items()]
        return "    <style>\n      <style-rule element='mark'>\n" + "\n".join(rules) + "\n      </style-rule>\n    </style>\n"

    def xml(self) -> str:
        hyper = build_extract(self.path, self.columns)
        cols = "\n".join(self.column_xml(c, "    ") for c in [*self.columns, *self.calcs])
        return f"""  <datasource caption={a(self.caption)} inline='true' name={a(self.name)} version='18.1'>
    <connection class='federated'>
      <named-connections>
        <named-connection caption={a(hyper.name)} name={a('hyper.' + self.key)}>
          <connection access_mode='readonly' authentication='auth-none' author-locale='en_US' class='hyper' dbname={a(hyper.as_posix())} default-settings='yes' schema='Extract' sslmode='' tablename='Extract' username='tableau_internal_user' />
        </named-connection>
      </named-connections>
      <relation connection={a('hyper.' + self.key)} name='Extract' table='[Extract].[Extract]' type='table' />
    </connection>
    <aliases enabled='yes' />
{cols}
    <layout dim-ordering='alphabetic' dim-percentage='0.5' measure-ordering='alphabetic' measure-percentage='0.4' show-structure='true' />
{self.style_xml()}  </datasource>"""


# ================================================================== worksheets
GENERATED = {"lat": "[Latitude (generated)]", "lon": "[Longitude (generated)]"}
DERIVATION = {"none": "None", "sum": "Sum", "avg": "Avg", "cnt": "Count", "usr": "User"}


@dataclass(frozen=True)
class F:
    """A field on a shelf: column name plus aggregation (none, sum, avg, cnt)."""
    col: str
    agg: str = "none"


def instance(ds: DataSource, f: F) -> tuple[str, str]:
    """(column-instance name, xml) for a field."""
    _, _, role, typ = ds.meta(f.col)
    if f.agg != "none" or role == "measure":
        typ, suffix = "quantitative", "qk"
    else:
        suffix = "ok" if typ == "ordinal" else "nk"
    name = f"[{f.agg}:{f.col}:{suffix}]"
    xml = f"        <column-instance column={a('[' + f.col + ']')} derivation={a(DERIVATION[f.agg])} name={a(name)} pivot='key' type={a(typ)} />"
    return name, xml


@dataclass
class Sheet:
    name: str
    ds: DataSource
    rows: list = dc_field(default_factory=list)
    cols: list = dc_field(default_factory=list)
    mark: str = "Bar"
    color: F | None = None
    text: F | None = None
    detail: list = dc_field(default_factory=list)
    tooltip: list = dc_field(default_factory=list)
    filters: dict = dc_field(default_factory=dict)          # col -> allowed values
    filter_group: int = 0                                   # shared filter group (for a dashboard filter control)
    sort_by: tuple | None = None                            # (dimension F, measure F, "ASC"|"DESC")
    manual_sort: dict = dc_field(default_factory=dict)      # col -> ordered values
    title: str = ""
    aggregate: bool = True
    seq_palette: str = ""                                   # named Tableau palette for a continuous Color
    label_format: str = ""                                  # number format of mark labels
    axis_format: str = ""                                   # number format of the quantitative axis
    mark_color: str = TAB_BLUE                              # used when nothing is on Color
    label_width: int = 0                                    # row-header width in pixels
    kpi: bool = False                                       # big-number tile

    def ref(self, f) -> str:
        if isinstance(f, str):
            return f"[{self.ds.name}].{GENERATED[f]}"
        return f"[{self.ds.name}].{instance(self.ds, f)[0]}"

    def fields(self) -> list[F]:
        out = [*self.rows, *self.cols, *self.detail, *self.tooltip, *[x for x in (self.color, self.text) if x]]
        out += [F(c) for c in self.filters] + [F(c) for c in self.manual_sort]
        if self.sort_by:
            out += [self.sort_by[0], self.sort_by[1]]
        return [f for f in out if isinstance(f, F)]

    def is_quant(self, f) -> bool:
        return isinstance(f, F) and (f.agg != "none" or self.ds.meta(f.col)[2] == "measure")

    def xml(self) -> str:
        ds = self.ds
        if self.seq_palette and self.color:
            ds.sequential[instance(ds, self.color)[0]] = self.seq_palette
        fields = list(dict.fromkeys(self.fields()))
        deps = [ds.column_xml(c, "        ") for c in dict.fromkeys(f.col for f in fields)]
        insts = list(dict.fromkeys(instance(ds, f)[1] for f in fields))

        filters = []
        for col, values in self.filters.items():
            inst = instance(ds, F(col))[0]
            members = [str(v) if ds.columns.get(col) == "integer" else '"' + str(v) + '"' for v in values]
            group = f" filter-group='{self.filter_group}'" if self.filter_group and col == "year" else ""
            if len(members) == 1:
                body = (f"        <groupfilter function='member' level={a(inst)} member={a(members[0])} "
                        "user:ui-domain='database' user:ui-enumeration='inclusive' user:ui-marker='enumerate' />")
            else:
                inner = "\n".join(f"          <groupfilter function='member' level={a(inst)} member={a(m)} />" for m in members)
                body = (f"        <groupfilter function='union' user:ui-domain='database' user:ui-enumeration='inclusive' "
                        f"user:ui-marker='enumerate'>\n{inner}\n        </groupfilter>")
            filters.append(f"      <filter class='categorical' column={a('[' + ds.name + '].' + inst)}{group}>\n{body}\n      </filter>")
        sorts = []
        for col, order in self.manual_sort.items():
            buckets = "\n".join(f"          <bucket>{escape(chr(34) + v + chr(34))}</bucket>" for v in order)
            sorts.append(f"      <sort class='manual' column={a(self.ref(F(col)))} direction='ASC'>\n        <dictionary>\n{buckets}\n"
                         f"        </dictionary>\n      </sort>")
        if self.sort_by:
            dim, meas, direction = self.sort_by
            sorts.append(f"      <sort class='computed' column={a(self.ref(dim))} direction={a(direction)} using={a(self.ref(meas))} />")

        rules = [f"      <style-rule element='worksheet'>\n        <format attr='font-family' value={a(B_FONT)} />\n      </style-rule>",
                 "      <style-rule element='gridline'>\n        <format attr='stroke-size' value='0' />\n      </style-rule>",
                 "      <style-rule element='zeroline'>\n        <format attr='stroke-size' value='0' />\n      </style-rule>"]
        if self.kpi and self.text:
            rules.append(f"      <style-rule element='cell'>\n        <format attr='font-family' value={a(H_FONT)} />\n"
                         f"        <format attr='font-size' value='26' />\n        <format attr='color' value={a(INK)} />\n      </style-rule>")
        if self.label_format and self.text:
            rules.append(f"      <style-rule element='cell'>\n        <format attr='text-format' field={a(self.ref(self.text))} value={a(self.label_format)} />\n      </style-rule>")
        quant = next((f for f in [*self.cols, *self.rows] if self.is_quant(f)), None)
        if self.axis_format and quant:
            rules.append(f"      <style-rule element='axis'>\n        <format attr='text-format' field={a(self.ref(quant))} value={a(self.axis_format)} />\n      </style-rule>")
        if self.label_width:
            for r in self.rows:
                if isinstance(r, F) and not self.is_quant(r):
                    rules.append(f"      <style-rule element='header'>\n        <format attr='width' field={a(self.ref(r))} value='{self.label_width}' />\n      </style-rule>")
        style = "    <style>\n" + "\n".join(rules) + "\n    </style>"

        enc = []
        if self.color:
            enc.append(f"            <color column={a(self.ref(self.color))} />")
        if self.text:
            enc.append(f"            <text column={a(self.ref(self.text))} />")
        enc += [f"            <lod column={a(self.ref(d))} />" for d in self.detail]
        enc += [f"            <tooltip column={a(self.ref(t))} />" for t in self.tooltip]
        encodings = ("          <encodings>\n" + "\n".join(enc) + "\n          </encodings>\n") if enc else ""
        pane_fmt = []
        if self.text and self.mark in ("Bar", "Square"):
            pane_fmt.append("              <format attr='mark-labels-show' value='true' />")
        if self.mark == "Circle":
            pane_fmt.append("              <format attr='size' value='2.2' />")
        if not self.color and self.mark in ("Bar", "Line", "Square", "Circle"):
            pane_fmt.append(f"              <format attr='mark-color' value={a(self.mark_color)} />")
        pane_style = ("          <style>\n            <style-rule element='mark'>\n" + "\n".join(pane_fmt)
                      + "\n            </style-rule>\n          </style>\n") if pane_fmt else ""
        if self.kpi:
            title = (f"  <layout-options>\n    <title>\n      <formatted-text>\n        <run fontcolor={a(MUTED)} fontname={a(B_FONT)} "
                     f"fontsize='10'>{escape(self.title.upper())}</run>\n      </formatted-text>\n    </title>\n  </layout-options>\n")
        elif self.title:
            title = (f"  <layout-options>\n    <title>\n      <formatted-text>\n        <run fontcolor={a(INK)} fontname={a(H_FONT)} "
                     f"fontsize='12'>{escape(self.title)}</run>\n      </formatted-text>\n    </title>\n  </layout-options>\n")
        else:
            title = ""
        rows = " / ".join(self.ref(r) for r in self.rows)
        cols = " / ".join(self.ref(c) for c in self.cols)
        mapsource = "      <mapsources>\n        <mapsource name='Tableau' />\n      </mapsources>\n" if self.mark == "Map" else ""
        mark = "Automatic" if self.mark == "Map" else self.mark
        return f"""<worksheet name={a(self.name)}>
{title}  <table>
    <view>
      <datasources>
        <datasource caption={a(ds.caption)} name={a(ds.name)} />
      </datasources>
{mapsource}      <datasource-dependencies datasource={a(ds.name)}>
{chr(10).join(deps)}
{chr(10).join(insts)}
      </datasource-dependencies>
{chr(10).join(filters + sorts)}
      <aggregation value={a('true' if self.aggregate else 'false')} />
    </view>
{style}
    <panes>
      <pane selection-relaxation-option='selection-relaxation-allow'>
        <view>
          <breakdown value='auto' />
        </view>
        <mark class={a('Text' if self.kpi else mark)} />
{encodings}{pane_style}      </pane>
    </panes>
    <rows>{escape(rows)}</rows>
    <cols>{escape(cols)}</cols>
  </table>
</worksheet>"""


def worksheet_window(name: str) -> str:
    return f"""    <window class='worksheet' hidden='true' name={a(name)}>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'>
            <card type='columns' />
          </strip>
          <strip size='2147483647'>
            <card type='rows' />
          </strip>
          <strip size='31'>
            <card type='title' />
          </strip>
        </edge>
      </cards>
    </window>"""


# ================================================================== dashboards and story
def zone_style(bg: str, margin: int = 0, padding: int = 0) -> str:
    return (f"          <zone-style>\n            <format attr='border-color' value='#000000' />\n"
            f"            <format attr='border-style' value='none' />\n            <format attr='border-width' value='0' />\n"
            f"            <format attr='margin' value='{margin}' />\n            <format attr='padding' value='{padding}' />\n"
            f"            <format attr='background-color' value={a(bg)} />\n          </zone-style>")


@dataclass
class Dashboard:
    name: str
    title: str
    subtitle: str
    kpis: list          # KPI sheet names, shown as a row of tiles
    layout: list        # (sheet name, x, y, w, h) in percent of the chart area
    footnote: str = "Source: HMDA via FFIEC/CFPB Data Browser API, New Jersey home-purchase applications. Analysis: github.com/hkandan-hash/nj-mortgage-analytics"
    filter_zone: tuple | None = None  # (sheet name, field ref, caption) for an on-dashboard filter control

    def xml(self, zone_id) -> str:
        # vertical bands, in 1/100000 of the dashboard
        header_h, kpi_h, foot_h = 12500, 12000, 4000
        chart_y = header_h + kpi_h + 1500
        chart_h = 100000 - chart_y - foot_h - 500
        zones = [f"""        <zone h='{header_h}' id='{zone_id()}' type-v2='text' w='100000' x='0' y='0'>
          <formatted-text>
            <run fontcolor={a(ON_DARK)} fontname={a(H_FONT)} fontsize='20'>{escape(self.title)}</run>
            <run>&#198;&#10;</run>
            <run fontcolor={a(ON_DARK_2)} fontname={a(L_FONT)} fontsize='11'>{escape(self.subtitle)}</run>
          </formatted-text>
{zone_style(NAVY, 0, 14)}
        </zone>"""]
        n = len(self.kpis)
        filter_w = 16000 if self.filter_zone else 0
        tile_w = (100000 - 3000 - filter_w - (n - 1) * 1000) // max(n, 1)
        for i, k in enumerate(self.kpis):
            zones.append(f"        <zone h='{kpi_h}' id='{zone_id()}' name={a(k)} show-title='true' w='{tile_w}' "
                         f"x='{1500 + i * (tile_w + 1000)}' y='{header_h + 1000}'>\n{zone_style(CARD, 0, 6)}\n        </zone>")
        if self.filter_zone:
            sheet, param, _ = self.filter_zone
            zones.append(f"        <zone h='{kpi_h}' id='{zone_id()}' mode='radiolist' name={a(sheet)} param={a(param)} "
                         f"type-v2='filter' values='database' w='{filter_w - 1000}' x='{100000 - 1500 - filter_w + 1000}' y='{header_h + 1000}'>\n"
                         f"{zone_style(CARD, 0, 6)}\n        </zone>")
        for sheet, x, y, w, h in self.layout:
            zx = 1500 + int(x * 970)
            zy = chart_y + int(y * chart_h / 100)
            zones.append(f"        <zone h='{int(h * chart_h / 100) - 1000}' id='{zone_id()}' name={a(sheet)} w='{int(w * 970) - 1000}' "
                         f"x='{zx}' y='{zy}'>\n{zone_style(CARD, 0, 10)}\n        </zone>")
        zones.append(f"""        <zone h='{foot_h}' id='{zone_id()}' type-v2='text' w='97000' x='1500' y='{100000 - foot_h}'>
          <formatted-text>
            <run fontcolor={a(MUTED)} fontname={a(B_FONT)} fontsize='8'>{escape(self.footnote)}</run>
          </formatted-text>
        </zone>""")
        return f"""    <dashboard name={a(self.name)}>
      <style>
        <style-rule element='dash-container'>
          <format attr='background-color' value={a(CANVAS)} />
        </style-rule>
      </style>
      <size maxheight='{H}' maxwidth='{W}' minheight='{H}' minwidth='{W}' sizing-mode='fixed' />
      <zones>
        <zone h='100000' id='{zone_id()}' type-v2='layout-basic' w='100000' x='0' y='0'>
{chr(10).join(zones)}
        </zone>
      </zones>
    </dashboard>"""

    def sheets(self) -> list[str]:
        return [*self.kpis, *(z[0] for z in self.layout)]


def story_xml(points: list[tuple[str, str]], zone_id) -> str:
    sp = "\n".join(f"                <story-point caption={a(cap)} captured-sheet={a(sheet)} id='{i + 1}' />"
                   for i, (sheet, cap) in enumerate(points))
    outer, flow, title, nav, flip = (zone_id() for _ in range(5))
    # The navigator (captions) and the flipboard (story content) must point at each other.
    return f"""    <dashboard name={a(STORY)} type='storyboard'>
      <style>
        <style-rule element='dash-container'>
          <format attr='background-color' value={a(CANVAS)} />
        </style-rule>
      </style>
      <size maxheight='{H + 150}' maxwidth='{W + 20}' minheight='{H + 150}' minwidth='{W + 20}' sizing-mode='fixed' />
      <zones>
        <zone h='100000' id='{outer}' type-v2='layout-basic' w='100000' x='0' y='0'>
          <zone h='100000' id='{flow}' param='vert' removable='false' type-v2='layout-flow' w='100000' x='0' y='0'>
          <zone h='4000' id='{title}' type-v2='title' w='100000' x='0' y='0' />
          <zone h='10000' id='{nav}' is-fixed='true' paired-zone-id='{flip}' removable='false' type-v2='flipboard-nav' w='100000' x='0' y='4000' />
          <zone h='86000' id='{flip}' paired-zone-id='{nav}' removable='false' type-v2='flipboard' w='100000' x='0' y='14000'>
            <flipboard active-id='1' nav-type='caption' show-nav-arrows='true'>
              <story-points>
{sp}
              </story-points>
            </flipboard>
          </zone>
          </zone>
        </zone>
      </zones>
    </dashboard>"""


def actions_xml(highlights: list[tuple[str, str, str]]) -> str:
    """Highlight actions: (dashboard, field caption, action caption). Selecting a mark highlights the
    same value in every sheet on that dashboard."""
    out = []
    for i, (dash, field_caption, caption) in enumerate(highlights, start=1):
        out.append(f"""    <action caption={a(caption)} name={a(f'[Action{i}_highlight]')}>
      <activation auto-clear='true' type='on-select' />
      <source dashboard={a(dash)} type='sheet' />
      <command command='tsc:brush'>
        <param name='field-captions' value={a(field_caption)} />
        <param name='target' value={a(dash)} />
      </command>
    </action>""")
    return "  <actions>\n" + "\n".join(out) + "\n  </actions>"


# ================================================================== content
SHORT_EFFECT = {
    "+10 points loan-to-value": "+10 pts loan-to-value", "+10 points debt-to-income": "+10 pts debt-to-income",
    "Income doubles": "Income doubles*", "Loan amount doubles": "Loan amount doubles",
    "+10 points tract minority share": "+10 pts tract minority",
    "+25 points tract income (% of area median)": "+25 pts tract income",
    "FHA (vs conventional)": "FHA loan", "VA (vs conventional)": "VA loan",
    "Investment property (vs own home)": "Investment property", "Second home (vs own home)": "Second home",
    "Jumbo (vs conforming)": "Jumbo loan", "No automated underwriting (vs DU)": "No AUS used",
    "Freddie Mac LPA (vs DU)": "Freddie Mac LPA",
}
FAIR_LABELS = {"actual_denial_rate": "Actual denial rate", "mean_predicted": "Average predicted risk",
               "false_positive_rate": "False positive rate", "false_negative_rate": "False negative rate"}


def build() -> str:
    reg = json.loads((METRICS_DIR / "regression.json").read_text())
    cls = json.loads((METRICS_DIR / "classification.json").read_text())
    best_cls = next(m for m in cls["models"] if m["model"] == cls["best_model"])
    best_reg = next(m for m in reg["models"] if m["model"] == reg["best_model"])
    base_cls = next(m for m in cls["models"] if m["model"].startswith("Baseline"))
    with (EXPORT_DIR / "county_summary.csv").open(encoding="utf-8") as f:
        counties = [r for r in csv.DictReader(f) if r["year"] == str(YEAR)]
    hi = max(counties, key=lambda r: float(r["denial_rate"]))
    lo = min(counties, key=lambda r: float(r["denial_rate"]))
    total = sum(int(r["applications"]) for r in counties)
    denied = sum(int(r["applications"]) * float(r["denial_rate"]) for r in counties)
    with (EXPORT_DIR / "denial_by_dti.csv").open(encoding="utf-8") as f:
        dti_rates = {r["dti_band"]: float(r["denial_rate"]) for r in csv.DictReader(f) if r["year"] == str(YEAR)}
    band_rate = dti_rates.__getitem__
    with (EXPORT_DIR / "denial_reasons.csv").open(encoding="utf-8") as f:
        top_reason = max((r for r in csv.DictReader(f) if r["year"] == str(YEAR)), key=lambda r: float(r["share"]))
    gaps = {g["race_ethnicity"]: g for g in reg["group_gaps"]}
    fr = {g["race_ethnicity"]: g for g in cls["fairness"]["race_ethnicity"]}
    hi_fpr = max(fr.values(), key=lambda g: g["false_positive_rate"])
    lo_fpr = min(fr.values(), key=lambda g: g["false_positive_rate"])
    eff = {SHORT_EFFECT.get(e["change"], e["change"]): e["effect"] for e in reg["effects"]}
    his, blk = gaps.get("Hispanic or Latino", {}), gaps.get("Black", {})

    # ---------------------------------------------------------------- derived tables
    effects_csv = write_csv("pricing_effects_labeled.csv", ["factor", "effect", "ci"],
                            [[SHORT_EFFECT.get(e["change"], e["change"]), e["effect"], e["ci"]] for e in reg["effects"]])
    fair_rows = []
    with (EXPORT_DIR / "fairness_audit.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["attribute"] == "race_ethnicity":
                fair_rows += [[r["race_ethnicity"], label, r[k], r["applications"]] for k, label in FAIR_LABELS.items()]
    fair_csv = write_csv("fairness_long.csv", ["group", "measure", "value", "applications"], fair_rows)
    model_rows = []
    for task, data, metric in (("Pricing (regression)", reg, "R2"), ("Denial (classification)", cls, "PR AUC")):
        for m in data["models"]:
            name = "Baseline" if m["model"].startswith("Baseline") else m["model"].replace("Median (quantile) regression", "Median regression")
            model_rows.append([task, name, "Best model" if m["model"] == data["best_model"] else "Alternative",
                               round(max(m[metric], 0), 4)])
    models_csv = write_csv("model_scores.csv", ["task", "model", "highlight", "score"], model_rows)
    # Statewide totals include applications with an unknown county, so take them from the dashboard data.
    dash_data = json.loads((ROOT / "docs" / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
    state = next(k for k in dash_data["kpis"] if k["year"] == YEAR)
    mid = [band_rate(b) for b in ["20-29%", "30-35%", "36-42%", "43-49%"]]
    kpi_rows = [
        ["apps", f"Applications, {YEAR}", f"{state['applications']:,}"],
        ["denial", f"Denial rate, {YEAR}", f"{state['denial_rate']:.1%}"],
        ["high", f"Highest: {hi['county']}", f"{float(hi['denial_rate']):.1%}"],
        ["low", f"Lowest: {lo['county']}", f"{float(lo['denial_rate']):.1%}"],
        ["dti_mid", "Denied, DTI 20-49%", f"{min(mid):.0%}-{max(mid):.0%}"],
        ["dti_50", "Denied, DTI 50-60%", f"{band_rate('50-60%'):.0%}"],
        ["dti_60", "Denied, DTI over 60%", f"{band_rate('>60%'):.0%}"],
        ["reason", "Denials citing DTI" if "Debt" in top_reason["denial_reason"] else f"Denials citing {top_reason['denial_reason'].lower()}", f"{float(top_reason['share']):.0%}"],
        ["r2", "Best pricing model R²", f"{best_reg['R2']:.2f}"],
        ["lender", "R² added by lender", f"+{reg['ols_r2_with_lender'] - reg['ols_r2_without_lender']:.2f}"],
        ["investor", "Investment property", f"{eff['Investment property']:+.2f} pts"],
        ["va", "VA loan", f"{eff['VA loan']:+.2f} pts"],
        ["auc", "ROC AUC", f"{best_cls['ROC AUC']:.3f}"],
        ["ap", "Average precision", f"{best_cls['PR AUC']:.3f}"],
        ["recall", "Denials caught", f"{best_cls['Recall']:.0%}"],
        ["precision", "Flags correct", f"{best_cls['Precision']:.0%}"],
        ["fpr_gap", "False-positive gap", f"{hi_fpr['false_positive_rate'] / lo_fpr['false_positive_rate']:.1f}x"],
        ["his_gap", "Hispanic pricing gap", f"{his.get('residual', 0) * 100:+.1f} bps"],
        ["blk_gap", "Black pricing gap", f"{blk.get('residual', 0) * 100:+.1f} bps"],
        ["inputs", "Protected inputs used", "0"],
    ]
    kpi_csv = write_csv("kpis.csv", ["key", "label", "display"], kpi_rows)
    imp_rows = []
    for task, data in (("Regression (rate spread)", reg), ("Classification (denial)", cls)):
        top = sorted(data["importance"], key=lambda r: -r["importance"])[:8]
        imp_rows += [[task, r["feature"], round(r["importance"], 4)] for r in top]
    importance_csv = write_csv("feature_importance_top8.csv", ["task", "feature", "importance"], imp_rows)

    # ---------------------------------------------------------------- data sources
    dti_order = ["<20%", "20-29%", "30-35%", "36-42%", "43-49%", "50-60%", ">60%"]
    ltv_order = ["≤60%", "60-80%", "80-90%", "90-95%", "95-97%", ">97%"]
    county = DataSource("county", EXPORT_DIR / "county_summary.csv", "County summary", dims={"year", "county_fips"},
                        calcs={"State": Calc("State", "string", "dimension", '"New Jersey"')},
                        semantic={"county": "[County].[Name]", "State": "[State].[Name]"},
                        captions={"county": "County", "denial_rate": "Denial rate", "median_income_k": "Median income ($k)",
                                  "applications": "Applications", "year": "Year"},
                        formats={"denial_rate": "p0.0%", "applications": "n#,##0", "median_income_k": "c\"$\"#,##0\"k\""})
    grid = DataSource("grid", EXPORT_DIR / "denial_by_dti_ltv.csv", "Denial by DTI and LTV", dims={"year"},
                      calcs={"Denial rate": Calc("Denial rate", "real", "measure", "IF [applications] >= 50 THEN [denial_rate] END")},
                      captions={"dti_band": "Debt-to-income", "ltv_band": "Loan-to-value", "applications": "Applications"},
                      formats={"Denial rate": "p0%", "applications": "n#,##0"})
    reasons = DataSource("reasons", EXPORT_DIR / "denial_reasons.csv", "Denial reasons", dims={"year"},
                         captions={"denial_reason": "Reason", "share": "Share of denials", "denials": "Denials"},
                         formats={"share": "p0%", "denials": "n#,##0"})
    models = DataSource("models", models_csv, "Model scores", captions={"model": "Model", "score": "Score", "highlight": "Highlight"},
                        formats={"score": "n0.000"})
    effects = DataSource("effects", effects_csv, "Pricing effects",
                         calcs={"Direction": Calc("Direction", "string", "dimension",
                                                  'IF [effect] > 0 THEN "Raises the price" ELSE "Lowers the price" END')},
                         captions={"factor": "Factor", "effect": "Change in rate spread (pct. points)", "ci": "95% interval (±)"},
                         formats={"effect": "n+0.00;-0.00", "ci": "n0.00"})
    importance = DataSource("importance", importance_csv, "Feature importance",
                            captions={"feature": "Feature", "importance": "Importance"}, formats={"importance": "n0.000"})
    roc = DataSource("roc", EXPORT_DIR / "roc_curves.csv", "ROC curves",
                     captions={"fpr": "False positive rate", "tpr": "True positive rate", "model": "Model"},
                     formats={"fpr": "p0%", "tpr": "p0%"})
    fair = DataSource("fair", fair_csv, "Fairness audit", captions={"value": "Rate", "group": "Group", "measure": "Measure"},
                      formats={"value": "p0.0%", "applications": "n#,##0"})
    gap = DataSource("gaps", EXPORT_DIR / "pricing_gaps.csv", "Pricing gaps",
                     calcs={"Unexplained gap (bps)": Calc("Unexplained gap (bps)", "real", "measure", "[residual] * 100"),
                            "Significance": Calc("Significance", "string", "dimension",
                                                 'IF ABS([residual]) > 1.96 * [residual_se] THEN "Significant (95%)" ELSE "Not significant" END')},
                     captions={"race_ethnicity": "Group", "loans": "Loans"},
                     formats={"Unexplained gap (bps)": "n+0.0;-0.0", "loans": "n#,##0"})
    apps = DataSource("apps", EXPORT_DIR / f"applications_{YEAR}_scored.csv", f"Applications {YEAR} (scored)", dims={"year", "county_fips"},
                      calcs={"Risk band": Calc("Risk band", "string", "dimension",
                                               'IF [denial_probability] >= 0.5 THEN "High 50%+" '
                                               'ELSEIF [denial_probability] >= 0.2 THEN "Elevated 20-50%" '
                                               'ELSEIF [denial_probability] >= 0.08 THEN "Moderate 8-20%" ELSE "Low <8%" END')},
                      captions={"denied": "Denial rate"}, formats={"denied": "p0%"})
    kpi = DataSource("kpi", kpi_csv, "Key figures", captions={"display": "Value", "label": "Label", "key": "Key"})
    sources = [county, grid, reasons, models, effects, importance, roc, fair, gap, apps, kpi]

    # ---------------------------------------------------------------- worksheets
    kpi_sheets = {k: Sheet(f"KPI {k}", kpi, text=F("display"), filters={"key": [k]}, title=label, kpi=True)
                  for k, label, _ in kpi_rows}
    year_filter = ("County map", f"[{county.name}].[none:year:ok]", "Year")
    sheets = [
        *kpi_sheets.values(),
        Sheet("County map", county, rows=["lat"], cols=["lon"], mark="Map", color=F("denial_rate", "avg"),
              seq_palette="blue_10_0", detail=[F("county"), F("State")], tooltip=[F("applications", "sum"), F("median_income_k", "avg")],
              filters={"year": [YEAR]}, filter_group=1, title="Denial rate by county (click a county)"),
        Sheet("County ranking", county, rows=[F("county")], cols=[F("denial_rate", "avg")], text=F("denial_rate", "avg"),
              tooltip=[F("applications", "sum")], filters={"year": [YEAR]}, filter_group=1,
              sort_by=(F("county"), F("denial_rate", "avg"), "DESC"), label_format="p0.0%", axis_format="p0%",
              label_width=100, title="Counties ranked by denial rate"),
        Sheet("DTI x LTV", grid, rows=[F("dti_band")], cols=[F("ltv_band")], mark="Square", color=F("Denial rate", "avg"),
              seq_palette="blue_10_0", text=F("Denial rate", "avg"), tooltip=[F("applications", "sum")], filters={"year": [YEAR]},
              manual_sort={"dti_band": dti_order, "ltv_band": ltv_order}, label_format="p0%", label_width=70,
              title="Denial rate: debt-to-income (rows) by loan-to-value (columns)"),
        Sheet("Denial reasons", reasons, rows=[F("denial_reason")], cols=[F("share", "sum")], text=F("share", "sum"),
              tooltip=[F("denials", "sum")], filters={"year": [YEAR]}, sort_by=(F("denial_reason"), F("share", "sum"), "DESC"),
              label_format="p0%", axis_format="p0%", label_width=180, title="Primary reason given for denial"),
        Sheet("Pricing effects", effects, rows=[F("factor")], cols=[F("effect", "sum")], color=F("Direction"),
              text=F("effect", "sum"), tooltip=[F("ci", "sum")], sort_by=(F("factor"), F("effect", "sum"), "DESC"),
              label_format="n+0.00;-0.00", axis_format="n+0.0;-0.0", label_width=190,
              title="Effect on price over market, all else equal (orange raises it, blue lowers it)"),
        Sheet("Pricing models", models, rows=[F("model")], cols=[F("score", "sum")], color=F("highlight"),
              text=F("score", "sum"), filters={"task": ["Pricing (regression)"]}, sort_by=(F("model"), F("score", "sum"), "DESC"),
              label_format="n0.00", axis_format="n0.0", label_width=180, title="Out-of-time R² by model"),
        Sheet("Pricing drivers", importance, rows=[F("feature")], cols=[F("importance", "sum")], filters={"task": ["Regression (rate spread)"]},
              sort_by=(F("feature"), F("importance", "sum"), "DESC"), axis_format="n0.00", label_width=140,
              title="Most important pricing factors"),
        Sheet("ROC curves", roc, rows=[F("tpr")], cols=[F("fpr")], mark="Line", color=F("model"), aggregate=False,
              axis_format="p0%", title="ROC curves (orange = gradient boosting)"),
        Sheet("Denial models", models, rows=[F("model")], cols=[F("score", "sum")], color=F("highlight"),
              text=F("score", "sum"), filters={"task": ["Denial (classification)"]}, sort_by=(F("model"), F("score", "sum"), "DESC"),
              label_format="n0.000", axis_format="n0.0", label_width=180,
              title=f"Average precision by model (random guess {base_cls['PR AUC']:.3f})"),
        Sheet("Risk vs outcome", apps, rows=[F("denied", "avg")], cols=[F("Risk band")], text=F("denied", "avg"),
              tooltip=[F("denied", "cnt")], manual_sort={"Risk band": ["Low <8%", "Moderate 8-20%", "Elevated 20-50%", "High 50%+"]},
              label_format="p0%", axis_format="p0%", title="Actual denial rate by predicted risk band"),
        Sheet("Denial drivers", importance, rows=[F("feature")], cols=[F("importance", "sum")], filters={"task": ["Classification (denial)"]},
              sort_by=(F("feature"), F("importance", "sum"), "DESC"), axis_format="n0.00", label_width=140,
              title="Most important denial factors"),
        Sheet("Fairness: denials", fair, rows=[F("group")], cols=[F("value", "sum")], mark="Circle", color=F("measure"),
              tooltip=[F("applications", "sum")], filters={"measure": ["Actual denial rate", "Average predicted risk"]},
              sort_by=(F("group"), F("value", "sum"), "DESC"), label_width=130,
              title="Denial rate: actual (blue) vs model's average predicted risk (orange)"),
        Sheet("Fairness: errors", fair, rows=[F("group")], cols=[F("value", "sum")], mark="Circle", color=F("measure"),
              tooltip=[F("applications", "sum")], filters={"measure": ["False positive rate", "False negative rate"]},
              sort_by=(F("group"), F("value", "sum"), "DESC"), label_width=130,
              title="Flag errors: missed denials (red) and wrongly flagged (teal)"),
        Sheet("Pricing gaps", gap, rows=[F("race_ethnicity")], cols=[F("Unexplained gap (bps)", "sum")], color=F("Significance"),
              text=F("Unexplained gap (bps)", "sum"), tooltip=[F("loans", "sum")], label_format="n+0.0;-0.0", axis_format="n+0;-0",
              label_width=140, title="Unexplained pricing gap, basis points (orange = statistically significant)"),
    ]

    # ---------------------------------------------------------------- dashboards
    dashboards = [
        Dashboard("1. Where", "Where New Jersey mortgage applications are denied",
                  "Raw county denial rates, not adjusted for applicant mix. Switch years; click a county to highlight it.",
                  ["KPI apps", "KPI denial", "KPI high", "KPI low"],
                  [("County map", 0, 0, 55, 100), ("County ranking", 55, 0, 45, 100)], filter_zone=year_filter),
        Dashboard("2. Why", "Denials take off once debt-to-income passes 50%",
                  "The steepest driver in the data and the most common reason lenders give. Blank cells: under 50 applications.",
                  ["KPI dti_mid", "KPI dti_50", "KPI dti_60", "KPI reason"],
                  [("DTI x LTV", 0, 0, 58, 100), ("Denial reasons", 58, 0, 42, 100)]),
        Dashboard("3. Pricing", "Lender choice and equity drive what borrowers pay over market",
                  "Price over market (rate spread). Trained on 2023, tested on 2024. *Income effect holds debt-to-income fixed.",
                  ["KPI r2", "KPI lender", "KPI investor", "KPI va"],
                  [("Pricing effects", 0, 0, 55, 100), ("Pricing models", 55, 0, 45, 50), ("Pricing drivers", 55, 50, 45, 50)]),
        Dashboard("4. Prediction", "Predicting denials a year ahead",
                  f"Trained on 2023, tested on {YEAR}. An early-outreach flag, not a credit decision.",
                  ["KPI auc", "KPI ap", "KPI recall", "KPI precision"],
                  [("ROC curves", 0, 0, 50, 55), ("Denial models", 50, 0, 50, 55), ("Risk vs outcome", 0, 55, 50, 45), ("Denial drivers", 50, 55, 50, 45)]),
        Dashboard("5. Fairness", "Most pricing gaps are explained; the flag's errors are uneven",
                  "Race is never a model input. No credit scores in HMDA, so gaps are not proof of bias. Click a group.",
                  ["KPI fpr_gap", "KPI his_gap", "KPI blk_gap", "KPI inputs"],
                  [("Fairness: denials", 0, 0, 50, 55), ("Fairness: errors", 50, 0, 50, 55), ("Pricing gaps", 0, 55, 100, 45)]),
    ]
    points = [
        ("1. Where", f"1 in 9 applications is denied; {hi['county']} {float(hi['denial_rate']):.1%} vs {lo['county']} {float(lo['denial_rate']):.1%}"),
        ("2. Why", "Denials take off above 50% debt-to-income"),
        ("3. Pricing", f"Investors pay {eff['Investment property']:+.2f} pts, VA {eff['VA loan']:+.2f}; lenders matter"),
        ("4. Prediction", f"The model catches {best_cls['Recall']:.0%} of denials, flagging {best_cls['Flagged']:.1%}"),
        ("5. Fairness", f"{his.get('residual', 0) * 100:+.1f} bps unexplained; flag errors {hi_fpr['false_positive_rate'] / lo_fpr['false_positive_rate']:.1f}x apart"),
    ]
    highlights = [("1. Where", "County", "Highlight county"), ("5. Fairness", "Group", "Highlight group")]

    active = os.environ.get("TABLEAU_ACTIVE", STORY)
    maxim = lambda n: " maximized='true'" if n == active else ""  # noqa: E731
    counter = iter(range(1, 100_000))
    zone_id = lambda: next(counter)  # noqa: E731
    sheet_xml = [s.xml() for s in sheets]  # first: sheets register palettes on their data sources
    dash_xml = "\n".join(dash.xml(zone_id) for dash in dashboards) + "\n" + story_xml(points, zone_id)
    windows = "\n".join([
        *(worksheet_window(s.name) for s in sheets),
        *(f"    <window class='dashboard'{maxim(dash.name)} name={a(dash.name)}>\n      <viewpoints>\n"
          + "\n".join(f"        <viewpoint name={a(s)}>\n          <zoom type='entire-view' />\n        </viewpoint>" for s in dash.sheets())
          + "\n      </viewpoints>\n      <active id='-1' />\n    </window>" for dash in dashboards),
        f"    <window class='dashboard'{maxim(STORY)} name={a(STORY)}>\n      <viewpoints />\n      <active id='-1' />\n    </window>",
    ])
    return f"""<?xml version='1.0' encoding='utf-8' ?>

<workbook original-version='18.1' source-build='2026.3.0' source-platform='win' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <datasources>
{chr(10).join(s.xml() for s in sources)}
  </datasources>
{actions_xml(highlights)}
  <worksheets>
{chr(10).join(sheet_xml)}
  </worksheets>
  <dashboards>
{dash_xml}
  </dashboards>
  <windows source-height='30'>
{windows}
  </windows>
</workbook>
"""


def package(xml: str) -> Path:
    """Write a .twbx: the workbook plus its Hyper extracts, with paths relative to the package,
    so it opens on any machine (the plain .twb points at this machine's extract folder)."""
    import zipfile

    target = OUT.with_suffix(".twbx")
    packaged = xml.replace(EXTRACT_DIR.as_posix() + "/", "Data/Extracts/")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(OUT.name, packaged)
        for hyper in sorted(EXTRACT_DIR.glob("*.hyper")):
            z.write(hyper, f"Data/Extracts/{hyper.name}")
    return target


if __name__ == "__main__":
    xml = build()
    OUT.write_text(xml, encoding="utf-8")
    twbx = package(xml)
    print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB) and", twbx, f"({twbx.stat().st_size / 1024 / 1024:.1f} MB)")
