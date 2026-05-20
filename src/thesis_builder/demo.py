"""Demo mode: in-memory FakeLLM + FakeSearch.

Lets you run the full real graph end-to-end with no API keys, no network.
Useful for the web UI, smoke tests, and architectural review.

Architecture is identical to a live run — the real Researcher, Skeptic,
Synthesizer, and LangGraph code execute. Only the leaf LLM calls and
web-search results are stubbed. Trace events still flow through the real
``TraceWriter`` so the metrics UI shows realistic-looking numbers.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from .llm import LLMClient, TraceWriter
from .schemas import (
    CoverageReport,
    Critique,
    FinalOutput,
    Node,
    NodeList,
    QueryPlan,
    SearchQuery,
    SearchResult,
    SourceSummary,
    Thesis,
)


# --- Canned content per theme ----------------------------------------------


@dataclass
class _ThemePack:
    theme: str
    queries: list[str]
    sources: list[tuple[str, str, str, list[str], list[str]]]  # title, url, snippet, layers, entities
    nodes: list[dict[str, Any]]
    investable_picks: list[str]
    headlines: list[str]
    summary: str
    diagram_mmd: str


def _ai_infra_pack() -> _ThemePack:
    queries = [
        "TSMC CoWoS advanced packaging capacity 2026",
        "HBM3e supply qualification Micron SK Hynix",
        "EUV lithography ASML High-NA timeline",
        "AI data center power capacity bottleneck Vertiv Eaton",
        "Nvidia Blackwell B200 supply chain risks",
        "AMD MI325X market share inference",
        "hyperscaler capex 2025 Microsoft Google Amazon Meta",
        "Broadcom AI custom silicon TPU pricing",
        "networking 800G optics Marvell Coherent",
        "data center liquid cooling SMCI Dell",
        "AI inference at edge Qualcomm Apple silicon",
        "Arm royalty model AI accelerator design wins",
        "TSMC 2nm GAA risk Samsung Intel foundry",
        "rare earths supply China export controls",
        "AI electricity demand grid interconnection queue",
        "wafer fab equipment Applied Materials KLA Lam",
        "AI training cluster economics OpenAI Anthropic",
        "fiber to data center long-haul Lumen Zayo",
    ]
    sources = [
        ("TSMC raises CoWoS capex amid AI demand", "https://www.example.com/ai/tsmc-cowos", "TSMC plans to roughly double monthly CoWoS capacity by end of 2025 to meet Nvidia and AMD orders.", ["enabling"], ["TSM", "NVDA", "AMD"]),
        ("Micron HBM3e qualified for Nvidia H200", "https://www.example.com/ai/micron-hbm", "Micron joined SK Hynix as a qualified HBM3e supplier; pricing remains tight through 2026.", ["enabling"], ["MU", "SK Hynix", "NVDA"]),
        ("ASML High-NA EUV ships first units", "https://www.example.com/ai/asml-high-na", "ASML began shipping High-NA EUV systems; Intel is the lead customer with TSMC ramping later.", ["enabling"], ["ASML", "INTC", "TSM"]),
        ("Vertiv guides up on AI cooling backlog", "https://www.example.com/ai/vertiv-q3", "Vertiv backlog hit a record on liquid-cooling and busway orders from US hyperscalers.", ["enabling"], ["VRT"]),
        ("Eaton books grid orders for data-center build-out", "https://www.example.com/ai/eaton-grid", "Eaton's electrical segment growth driven by 1-GW-class data center campuses in Virginia and Texas.", ["enabling"], ["ETN"]),
        ("Nvidia Blackwell yield update from CFO", "https://www.example.com/ai/nvda-blackwell", "Blackwell yields normalized after early thermal issues; B200 contributing meaningfully to FY26.", ["integrators"], ["NVDA"]),
        ("AMD MI325X wins at Microsoft and Meta", "https://www.example.com/ai/amd-mi325", "AMD's MI325X secured deployments at Microsoft Azure and Meta; ROCm software gap narrowing.", ["integrators"], ["AMD", "MSFT", "META"]),
        ("Hyperscaler capex hits $200B annualized run rate", "https://www.example.com/ai/capex", "Combined capex from MSFT, GOOGL, AMZN, META is on a $200B+ pace, most directed at AI infra.", ["applications"], ["MSFT", "GOOGL", "AMZN", "META"]),
        ("Broadcom TPU revenue surprise", "https://www.example.com/ai/avgo-tpu", "Broadcom's custom-silicon AI revenue (Google TPU + Meta MTIA) now a billion-dollar quarterly line.", ["integrators"], ["AVGO", "GOOGL", "META"]),
        ("Marvell 1.6T optical DSP shipping", "https://www.example.com/ai/mrvl-1.6t", "Marvell's 1.6T DSP for AI back-end fabric ramped ahead of plan; Arista, Cisco are buyers.", ["enabling"], ["MRVL", "ANET", "CSCO"]),
        ("Coherent VCSEL capacity for AI optics", "https://www.example.com/ai/cohr-vcsel", "Coherent expanding VCSEL fab to address 800G/1.6T optics demand for inter-rack interconnects.", ["enabling"], ["COHR"]),
        ("Supermicro signs liquid-cooling deal with Lambda", "https://www.example.com/ai/smci-lambda", "Supermicro liquid-cooled rack systems shipped to neo-cloud Lambda for inference workloads.", ["integrators"], ["SMCI"]),
        ("Qualcomm AI100 Ultra targets enterprise inference", "https://www.example.com/ai/qcom-ai100", "Qualcomm's AI100 Ultra accelerator sampling with hyperscaler and enterprise customers.", ["applications"], ["QCOM"]),
        ("Apple Private Cloud Compute runs on M-series", "https://www.example.com/ai/aapl-pcc", "Apple disclosed its server-class M-series silicon powering Apple Intelligence cloud workloads.", ["integrators", "applications"], ["AAPL"]),
        ("Arm hikes royalty rates for v9 cores in AI", "https://www.example.com/ai/arm-royalty", "Arm raised per-core royalty for v9 designs used in AI accelerators and inference SoCs.", ["enabling"], ["ARM"]),
        ("Samsung loses TSMC AI customer share", "https://www.example.com/ai/samsung-foundry", "Samsung Foundry's AI customer pipeline shrank; yield gaps at 3nm vs TSMC remain.", ["enabling"], ["Samsung", "TSM"]),
        ("Intel Foundry signs AI customer at 18A", "https://www.example.com/ai/intc-18a", "Intel Foundry announced an unnamed AI accelerator customer on 18A starting 2026.", ["enabling"], ["INTC"]),
        ("China rare-earth export curbs hit magnet supply", "https://www.example.com/ai/rare-earth", "China tightened export licensing for samarium and dysprosium, raising costs for AI server fans/motors.", ["raw_inputs"], ["China"]),
        ("Virginia data center interconnection queue at 30 GW", "https://www.example.com/ai/dominion-queue", "Dominion Energy's queue for data center connections passed 30 GW, mostly AI-driven.", ["raw_inputs", "enabling"], ["Dominion", "D"]),
        ("Applied Materials guides up on AI fab spending", "https://www.example.com/ai/amat", "Applied Materials cited AI-driven leading-edge fab spend as the primary growth vector.", ["enabling"], ["AMAT"]),
        ("KLA process control wins at advanced packaging", "https://www.example.com/ai/klac-packaging", "KLA's inspection tools gain share at TSMC's CoWoS lines.", ["enabling"], ["KLAC", "TSM"]),
        ("Lam Research etch share in 3D-NAND and HBM", "https://www.example.com/ai/lrcx-hbm", "Lam noted HBM stack etch as the fastest-growing application within its memory franchise.", ["enabling"], ["LRCX"]),
        ("Anthropic Claude training cluster on AWS Trainium2", "https://www.example.com/ai/anthropic-trainium", "Anthropic disclosed training Claude on Trainium2 capacity at scale within AWS.", ["applications"], ["AMZN", "Anthropic"]),
        ("OpenAI Stargate site selected in Abilene", "https://www.example.com/ai/openai-stargate", "OpenAI / Oracle / SoftBank Stargate joint venture broke ground in Texas.", ["applications"], ["ORCL", "OpenAI", "SoftBank"]),
        ("xAI Colossus 200k-GPU cluster online", "https://www.example.com/ai/xai-colossus", "xAI brought Colossus cluster online in Memphis using Nvidia H100/H200.", ["applications"], ["xAI", "NVDA"]),
        ("Coreweave Q3 commitments back-loaded", "https://www.example.com/ai/crwv", "CoreWeave reported $14B in remaining performance obligations, ramping into 2025-26.", ["integrators"], ["CRWV"]),
        ("Lumen wins $5B AI fiber buildout from Microsoft", "https://www.example.com/ai/lumen-msft", "Lumen secured a multi-billion dollar long-haul fiber build for Microsoft AI training.", ["enabling"], ["LUMN", "MSFT"]),
        ("PJM capacity auction clears at record price", "https://www.example.com/ai/pjm-auction", "PJM capacity auction cleared 10x prior year, reflecting AI data center load growth.", ["raw_inputs"], ["PJM"]),
        ("Constellation signs nuclear PPA with Microsoft", "https://www.example.com/ai/ceg-msft", "Constellation Energy will restart Three Mile Island Unit 1 under a 20-year MSFT PPA.", ["raw_inputs"], ["CEG", "MSFT"]),
        ("Schneider Electric data center revenue +30%", "https://www.example.com/ai/su-paris", "Schneider Electric's data center segment grew 30%, citing AI as primary driver.", ["enabling"], ["Schneider"]),
        ("Astera Labs ramps CXL retimers", "https://www.example.com/ai/alab", "Astera Labs reported strong ramp of PCIe retimers and CXL switches for AI servers.", ["enabling"], ["ALAB"]),
        ("Credo SerDes wins at hyperscaler ToR switch", "https://www.example.com/ai/crdo", "Credo Technology won an active electrical cable design at a US hyperscaler.", ["enabling"], ["CRDO"]),
        ("Nvidia networking (Mellanox) grows 50% YoY", "https://www.example.com/ai/nvda-mlnx", "Nvidia's networking segment, anchored by Spectrum-X and InfiniBand, grew 50% YoY.", ["integrators", "enabling"], ["NVDA"]),
        ("Equinix xScale JV adds AI-ready capacity", "https://www.example.com/ai/eqix-xscale", "Equinix and partners committed $15B to xScale hyperscale capacity.", ["enabling"], ["EQIX"]),
        ("Digital Realty leases up at record pace", "https://www.example.com/ai/dlr", "Digital Realty's leasing volume hit a record on AI tenant demand.", ["enabling"], ["DLR"]),
    ]
    nodes = [
        {"name": "Critical minerals (Cu, REEs, samarium)", "layer": "raw_inputs", "description": "Copper for switchgear, rare earths for high-density motors and AI server fans.", "public_names": ["FCX"], "private_names": [], "catalysts": ["China export curbs tighten REE supply"], "risks": ["Substitution to ferrite magnets in some applications"], "sources": ["https://www.example.com/ai/rare-earth"], "confidence": 0.55},
        {"name": "Power generation & PPAs", "layer": "raw_inputs", "description": "Nuclear and gas PPAs underwriting hyperscaler load growth.", "public_names": ["CEG", "D"], "private_names": [], "catalysts": ["Three Mile Island restart 2028", "PJM auction 10x clearing price"], "risks": ["Interconnection queues slip"], "sources": ["https://www.example.com/ai/ceg-msft", "https://www.example.com/ai/pjm-auction"], "confidence": 0.75},
        {"name": "Grid interconnection capacity", "layer": "raw_inputs", "description": "Transmission and substation buildout gating data center site readiness.", "public_names": ["D"], "private_names": [], "catalysts": ["Virginia queue at 30 GW"], "risks": ["Permit delays in PJM"], "sources": ["https://www.example.com/ai/dominion-queue"], "confidence": 0.6},
        {"name": "EUV lithography", "layer": "enabling", "description": "ASML monopoly on EUV and High-NA EUV is the deepest chokepoint in the chain.", "public_names": ["ASML"], "private_names": [], "catalysts": ["High-NA shipments to Intel, TSMC ramp"], "risks": ["Geopolitical export restrictions"], "sources": ["https://www.example.com/ai/asml-high-na"], "confidence": 0.92},
        {"name": "Advanced packaging (CoWoS)", "layer": "enabling", "description": "TSMC CoWoS capacity is the binding constraint for Blackwell-class accelerators.", "public_names": ["TSM"], "private_names": [], "catalysts": ["CoWoS capacity 2x by EOY 2025"], "risks": ["Yield on next-gen substrates"], "sources": ["https://www.example.com/ai/tsmc-cowos"], "confidence": 0.9},
        {"name": "HBM memory", "layer": "enabling", "description": "SK Hynix is the volume leader; Micron is the marginal supplier with the largest upside.", "public_names": ["MU"], "private_names": ["SK Hynix"], "catalysts": ["Micron qualified at Nvidia"], "risks": ["Samsung qualification could pressure pricing"], "sources": ["https://www.example.com/ai/micron-hbm"], "confidence": 0.82},
        {"name": "Networking silicon & optics", "layer": "enabling", "description": "Switch ASICs and DSP/VCSEL optics; tightly coupled to scale-up cluster sizes.", "public_names": ["AVGO", "MRVL", "COHR"], "private_names": [], "catalysts": ["1.6T optics ramp"], "risks": ["Co-packaged optics displaces pluggables"], "sources": ["https://www.example.com/ai/mrvl-1.6t", "https://www.example.com/ai/cohr-vcsel"], "confidence": 0.78},
        {"name": "Power, cooling, electrical", "layer": "enabling", "description": "Vertiv, Eaton, Schneider — picks-and-shovels for 1-GW data center campuses.", "public_names": ["VRT", "ETN"], "private_names": ["Schneider Electric"], "catalysts": ["Record backlogs on liquid cooling"], "risks": ["Cooling tech standardizing reduces margin"], "sources": ["https://www.example.com/ai/vertiv-q3", "https://www.example.com/ai/eaton-grid"], "confidence": 0.83},
        {"name": "Accelerator design", "layer": "integrators", "description": "Nvidia dominates training; AMD gaining inference; Broadcom owns the hyperscaler ASIC slot.", "public_names": ["NVDA", "AMD", "AVGO"], "private_names": [], "catalysts": ["MI325X design wins", "Blackwell ramp"], "risks": ["ASIC TCO undercuts merchant GPU at scale"], "sources": ["https://www.example.com/ai/nvda-blackwell", "https://www.example.com/ai/amd-mi325", "https://www.example.com/ai/avgo-tpu"], "confidence": 0.85},
        {"name": "Servers & systems integration", "layer": "integrators", "description": "Supermicro and Dell ship AI racks; CoreWeave / neo-clouds buy them.", "public_names": ["SMCI", "DELL", "CRWV"], "private_names": ["Lambda"], "catalysts": ["Liquid-cooled rack ramp"], "risks": ["Margin compression as ODMs commoditize"], "sources": ["https://www.example.com/ai/smci-lambda", "https://www.example.com/ai/crwv"], "confidence": 0.68},
        {"name": "Data center real estate", "layer": "enabling", "description": "Equinix and Digital Realty hold the prime metros; xScale capacity is pre-leased.", "public_names": ["EQIX", "DLR"], "private_names": [], "catalysts": ["Record leasing volumes"], "risks": ["Power-constrained markets cap growth"], "sources": ["https://www.example.com/ai/eqix-xscale", "https://www.example.com/ai/dlr"], "confidence": 0.72},
        {"name": "Hyperscaler model hosting", "layer": "applications", "description": "MSFT, GOOGL, AMZN, META capture the bulk of training and inference spend.", "public_names": ["MSFT", "GOOGL", "AMZN", "META"], "private_names": ["OpenAI", "Anthropic"], "catalysts": ["$200B+ annualized capex"], "risks": ["ROI on AI capex remains unproven"], "sources": ["https://www.example.com/ai/capex", "https://www.example.com/ai/anthropic-trainium", "https://www.example.com/ai/openai-stargate"], "confidence": 0.7},
        {"name": "Inference at the edge", "layer": "applications", "description": "Apple Silicon and Qualcomm AI100 push inference onto devices and private clouds.", "public_names": ["AAPL", "QCOM"], "private_names": [], "catalysts": ["Apple Private Cloud Compute"], "risks": ["Cloud inference TCO falls faster than expected"], "sources": ["https://www.example.com/ai/aapl-pcc", "https://www.example.com/ai/qcom-ai100"], "confidence": 0.6},
    ]
    picks = [
        "ASML — EUV monopoly is the deepest chokepoint; High-NA cycle extends moat into the next decade.",
        "TSM — CoWoS capacity is the binding constraint on AI accelerator supply; lock-in across NVDA, AMD, AVGO.",
        "MU — marginal HBM supplier with the largest unit-economic delta as supply tightens through 2026.",
        "VRT — liquid cooling and busway capture every 1-GW campus build, with record backlog visibility.",
        "AVGO — owns the hyperscaler custom-silicon slot (TPU + MTIA) plus the switch-ASIC franchise.",
    ]
    headlines = [
        "Accelerator economics: merchant GPU margins compress as ASIC TCO improves at scale.",
        "Power, not silicon, is the binding constraint for the next leg of AI capex.",
        "HBM is a 2-supplier oligopoly today, a 3-supplier market by late 2025 — pricing fades.",
        "Hyperscaler ROI on AI capex is still unproven; a single quarter of pullback re-rates the chain.",
        "Geopolitical export controls on EUV, HBM, and rare earths could fork the global supply chain.",
        "Edge inference (Apple, Qualcomm) cannibalizes a slice of cloud demand if device-side models hit quality.",
    ]
    summary = (
        "AI compute capex flows through three tangible chokepoints: ASML's EUV monopoly, TSMC's "
        "CoWoS packaging capacity, and the HBM oligopoly anchored by SK Hynix and Micron. Power and "
        "cooling (Vertiv, Eaton) are catching up in importance as 1-GW campuses become the unit of "
        "scale. The durable owners are the picks-and-shovels; accelerator margins (NVDA, AMD) face "
        "real ASIC competition from Broadcom's hyperscaler franchise. The bear case is straightforward "
        "— hyperscaler capex is running ahead of any clear AI revenue, and the chain is one capex "
        "guide-down away from a sharp re-rating. We hedge by skewing picks toward bottlenecks that "
        "would still bind if growth merely persists rather than accelerates."
    )
    diagram = """graph LR
  subgraph raw[Raw inputs]
    r1["Critical minerals<br/>FCX"]
    r2["Power & PPAs<br/>CEG, D"]
    r3["Grid interconnect<br/>Dominion"]
  end
  subgraph enabling[Enabling]
    e1["EUV lithography<br/>ASML"]
    e2["Adv. packaging<br/>TSM"]
    e3["HBM memory<br/>MU, SK Hynix"]
    e4["Networking silicon<br/>AVGO, MRVL"]
    e5["Power & cooling<br/>VRT, ETN"]
    e6["DC real estate<br/>EQIX, DLR"]
  end
  subgraph integrators[Integrators]
    i1["Accelerator design<br/>NVDA, AMD, AVGO"]
    i2["Systems & neo-clouds<br/>SMCI, CRWV"]
  end
  subgraph applications[Applications]
    a1["Hyperscaler hosting<br/>MSFT, GOOGL, AMZN"]
    a2["Edge inference<br/>AAPL, QCOM"]
  end
  r1 --> e2
  r2 --> e6
  r3 --> e6
  e1 --> e2 --> i1
  e3 --> i1
  e4 --> i1
  e5 --> i2
  e6 --> i2
  i1 --> a1
  i1 --> a2
  i2 --> a1"""
    return _ThemePack("AI infrastructure", queries, sources, nodes, picks, headlines, summary, diagram)


def _robotics_pack() -> _ThemePack:
    queries = [
        "harmonic drive strain wave gearing market share Nidec",
        "humanoid robot bill of materials Tesla Optimus Figure",
        "Cognex machine vision market share factory automation",
        "Fanuc ABB Yaskawa industrial robot orders 2026",
        "Boston Dynamics Stretch warehouse deployments",
        "Hesai Luminar lidar robotaxi BOM",
        "Nvidia Isaac robotics platform adoption",
        "Symbotic warehouse automation orders Walmart",
        "Intuitive Surgical da Vinci 5 launch",
        "Amazon robotics fulfillment capex",
        "ABB AMR Mobile robot acquisition strategy",
        "China humanoid robot Unitree Xiaomi",
        "force-torque sensor supply chain ATI",
        "robot dexterity end-effector startups",
        "Boston Dynamics Atlas commercialization",
        "rare earth magnet supply for robotic motors",
    ]
    sources = [
        ("Tesla Optimus BOM analysis", "https://www.example.com/r/optimus-bom", "Optimus uses ~28 actuators with harmonic drives and rare-earth magnets; supplier mix biased to Asia.", ["raw_inputs", "enabling"], ["TSLA"]),
        ("Harmonic Drive cycle inventory normalizes", "https://www.example.com/r/harmonic", "Harmonic Drive Systems guided to recovery in industrial channel and humanoid prototypes.", ["enabling"], ["Harmonic Drive"]),
        ("Nidec invests in humanoid motor capacity", "https://www.example.com/r/nidec", "Nidec announced a new high-density motor line aimed at humanoid robotics OEMs.", ["enabling"], ["Nidec"]),
        ("Cognex AI vision platform refresh", "https://www.example.com/r/cgnx", "Cognex unveiled an AI-first vision platform; deep-learning inspection wins at battery makers.", ["enabling"], ["CGNX"]),
        ("Fanuc orders trough behind us", "https://www.example.com/r/fanuc", "Fanuc executives signaled an order-cycle inflection in CNC and articulated robots.", ["integrators"], ["Fanuc"]),
        ("ABB acquires Sevensense for AMR vision", "https://www.example.com/r/abb-amr", "ABB bought Sevensense to integrate visual SLAM into its mobile robot stack.", ["integrators"], ["ABB"]),
        ("Boston Dynamics Stretch at GXO", "https://www.example.com/r/bdyn-gxo", "GXO Logistics expanded Stretch deployments to additional US warehouses.", ["applications", "integrators"], ["Boston Dynamics", "GXO"]),
        ("Symbotic backlog crosses $23B", "https://www.example.com/r/sym", "Symbotic reported $23B+ in remaining performance obligations, dominated by Walmart.", ["applications", "integrators"], ["SYM", "WMT"]),
        ("Intuitive Surgical da Vinci 5 ramp", "https://www.example.com/r/isrg", "ISRG's da Vinci 5 system shipments accelerated; instruments and accessories outpaced systems.", ["applications"], ["ISRG"]),
        ("Hesai wins automotive lidar program", "https://www.example.com/r/hsai", "Hesai disclosed a new lead lidar program at a Chinese EV OEM for L2++ ADAS.", ["enabling"], ["HSAI"]),
        ("Luminar refocuses on Volvo platform", "https://www.example.com/r/lazr", "Luminar restructured operations and narrowed focus to the Volvo EX90 program.", ["enabling"], ["LAZR"]),
        ("Nvidia Isaac platform expands GR00T models", "https://www.example.com/r/isaac", "Nvidia added new foundation models to Isaac/GR00T for humanoid pretraining.", ["enabling", "applications"], ["NVDA"]),
        ("Amazon robotics deployment hits 750k", "https://www.example.com/r/amzn-robots", "Amazon disclosed it operates more than 750,000 mobile robots in fulfillment.", ["applications"], ["AMZN"]),
        ("Sony semiconductor image sensor share", "https://www.example.com/r/sony-cis", "Sony retains >40% share in mobile and industrial CMOS image sensors.", ["enabling"], ["Sony"]),
        ("Keyence quarterly margins hold above 50%", "https://www.example.com/r/keyence", "Keyence reported sustained operating margins above 50% on factory automation sensors.", ["enabling"], ["Keyence"]),
        ("Unitree humanoid pricing pressures west", "https://www.example.com/r/unitree", "Unitree G1 humanoid priced sub-$20k undercuts US/EU OEM cost stacks.", ["integrators"], ["Unitree"]),
        ("Tesla Optimus production targets", "https://www.example.com/r/optimus-target", "Tesla reiterated multi-thousand-unit Optimus build in 2025, scaling to 50k+ in 2026.", ["integrators"], ["TSLA"]),
        ("Figure deal with BMW for body shop", "https://www.example.com/r/figure-bmw", "Figure AI deployed humanoids at BMW Spartanburg for material handling in body shop.", ["integrators", "applications"], ["Figure", "BMW"]),
        ("Yaskawa announces motoman M-series", "https://www.example.com/r/yaskawa", "Yaskawa launched compact M-series robots for cell-manufacturing applications.", ["integrators"], ["Yaskawa"]),
        ("China REE export curbs affect motors", "https://www.example.com/r/ree-motors", "China's export licensing for dysprosium pressures motor cost stacks for Western OEMs.", ["raw_inputs"], ["China"]),
    ]
    nodes = [
        {"name": "Rare-earth magnets & specialty materials", "layer": "raw_inputs", "description": "Dysprosium/samarium for high-density permanent-magnet motors used in actuators.", "public_names": [], "private_names": [], "catalysts": ["China export curbs"], "risks": ["Ferrite substitution"], "sources": ["https://www.example.com/r/ree-motors", "https://www.example.com/r/optimus-bom"], "confidence": 0.6},
        {"name": "Precision gearing (harmonic, cycloidal)", "layer": "enabling", "description": "Strain-wave reducers are the chokepoint for high-DOF robot joints.", "public_names": [], "private_names": ["Harmonic Drive Systems"], "catalysts": ["Humanoid prototypes pull demand"], "risks": ["New gearing designs from US/EU startups"], "sources": ["https://www.example.com/r/harmonic"], "confidence": 0.78},
        {"name": "Motors & actuators", "layer": "enabling", "description": "Nidec and Maxon supply high-density motors for industrial and humanoid platforms.", "public_names": [], "private_names": ["Nidec", "Maxon", "Moog"], "catalysts": ["Humanoid build ramp"], "risks": ["Vertical integration by TSLA/Figure"], "sources": ["https://www.example.com/r/nidec"], "confidence": 0.7},
        {"name": "Vision & sensing", "layer": "enabling", "description": "Cognex (industrial), Keyence (factory), Sony (CIS), Hesai/Luminar (lidar).", "public_names": ["CGNX", "HSAI", "LAZR"], "private_names": ["Keyence", "Sony"], "catalysts": ["AI-first vision systems"], "risks": ["Commoditization of lidar"], "sources": ["https://www.example.com/r/cgnx", "https://www.example.com/r/keyence", "https://www.example.com/r/hsai"], "confidence": 0.74},
        {"name": "Compute & foundation models", "layer": "enabling", "description": "Nvidia Isaac/GR00T as the platform layer for humanoid policy pretraining.", "public_names": ["NVDA"], "private_names": [], "catalysts": ["GR00T model releases"], "risks": ["Open-source alternatives"], "sources": ["https://www.example.com/r/isaac"], "confidence": 0.66},
        {"name": "Industrial robot OEMs", "layer": "integrators", "description": "Fanuc, ABB, Yaskawa, KUKA — incumbent franchise with cycle leverage.", "public_names": [], "private_names": ["Fanuc", "ABB", "Yaskawa", "KUKA"], "catalysts": ["Cycle inflection"], "risks": ["China onshoring share gains"], "sources": ["https://www.example.com/r/fanuc", "https://www.example.com/r/abb-amr", "https://www.example.com/r/yaskawa"], "confidence": 0.7},
        {"name": "Humanoids", "layer": "integrators", "description": "Tesla Optimus, Figure, Boston Dynamics, Unitree; capability-cost frontier is moving fast.", "public_names": ["TSLA"], "private_names": ["Figure", "Boston Dynamics", "Unitree"], "catalysts": ["BMW Figure deployment", "Optimus 2025 build"], "risks": ["Demonstrations vs. revenue gap"], "sources": ["https://www.example.com/r/figure-bmw", "https://www.example.com/r/optimus-target", "https://www.example.com/r/unitree"], "confidence": 0.55},
        {"name": "Warehouse automation", "layer": "applications", "description": "Symbotic (Walmart), Amazon robotics, Boston Dynamics Stretch at GXO.", "public_names": ["SYM", "AMZN"], "private_names": ["Boston Dynamics"], "catalysts": ["Symbotic $23B backlog"], "risks": ["Customer concentration (Walmart)"], "sources": ["https://www.example.com/r/sym", "https://www.example.com/r/amzn-robots", "https://www.example.com/r/bdyn-gxo"], "confidence": 0.7},
        {"name": "Surgical robotics", "layer": "applications", "description": "Intuitive Surgical's da Vinci 5 cycle plus razor/blade instrument economics.", "public_names": ["ISRG"], "private_names": [], "catalysts": ["da Vinci 5 ramp"], "risks": ["Competition from Medtronic Hugo"], "sources": ["https://www.example.com/r/isrg"], "confidence": 0.78},
        {"name": "Mobility / robotaxi", "layer": "applications", "description": "Autonomous-vehicle programs depend on sensor + compute pipelines from same supply chain.", "public_names": ["TSLA"], "private_names": ["Waymo", "Cruise"], "catalysts": ["Lidar BOM cost decline"], "risks": ["Regulation in US/EU"], "sources": ["https://www.example.com/r/hsai", "https://www.example.com/r/lazr"], "confidence": 0.5},
    ]
    picks = [
        "ISRG — best risk-adjusted exposure to robotics today: monopoly razor/blade economics in surgical automation.",
        "SYM — concentrated Walmart bet, but the backlog is real and the tech is in production.",
        "CGNX — picks-and-shovels for any robotics build-out; cycle inflection ahead.",
        "NVDA — Isaac/GR00T compounds the AI-infra bet into the robotics stack.",
        "HSAI — lidar BOM owner with diversified Chinese OEM wins; cheap on volume scaling.",
    ]
    headlines = [
        "Humanoid demos remain demos; revenue inflection is still 2027+ on any honest analysis.",
        "Industrial robotics is cyclical first, secular second — don't pay for secular at cycle highs.",
        "Vertical integration risk: TSLA and Figure could absorb most actuator value internally.",
        "China supplies the magnets, the motors, and increasingly the humanoid platforms themselves.",
        "Lidar is commoditizing — diversification across programs matters more than tech leadership.",
    ]
    summary = (
        "Robotics is a barbell: surgical and warehouse automation are real businesses today (ISRG, "
        "SYM, Amazon's internal fleet), while humanoids remain narrative-driven. The picks-and-shovels "
        "layer (vision, gearing, motors) is the most defensible exposure for the next 24 months. "
        "Vertical integration by Tesla and Figure is the biggest threat to the actuator suppliers; "
        "China's dominance of the magnet/motor chain is the biggest macro risk."
    )
    diagram = """graph LR
  subgraph raw[Raw inputs]
    r1["Rare-earth magnets<br/>(China supply)"]
  end
  subgraph enabling[Enabling]
    e1["Precision gearing<br/>Harmonic Drive"]
    e2["Motors & actuators<br/>Nidec, Maxon"]
    e3["Vision & sensing<br/>CGNX, Keyence, HSAI"]
    e4["Compute / models<br/>NVDA Isaac"]
  end
  subgraph integrators[Integrators]
    i1["Industrial OEMs<br/>Fanuc, ABB, Yaskawa"]
    i2["Humanoids<br/>TSLA, Figure, Unitree"]
  end
  subgraph applications[Applications]
    a1["Warehouse<br/>SYM, AMZN"]
    a2["Surgical<br/>ISRG"]
    a3["Mobility<br/>TSLA, Waymo"]
  end
  r1 --> e2
  e1 --> i2
  e2 --> i1
  e2 --> i2
  e3 --> i1
  e3 --> a3
  e4 --> i2
  i1 --> a1
  i2 --> a1
  i2 --> a2"""
    return _ThemePack("Robotics", queries, sources, nodes, picks, headlines, summary, diagram)


def _energy_pack() -> _ThemePack:
    queries = [
        "lithium price 2026 Albemarle SQM",
        "copper supply deficit smelter capacity",
        "battery cell capacity CATL LG Energy",
        "QuantumScape solid-state battery commercialization",
        "First Solar utility-scale module pricing",
        "Enphase microinverter US residential",
        "grid transmission Quanta PWR backlog",
        "EV charging infrastructure ChargePoint EVgo",
        "heat pump adoption Carrier Trane Lennox",
        "wind turbine OEM Vestas Siemens Gamesa",
        "nuclear SMR NuScale Holtec Oklo",
        "hydrogen electrolyzer Plug Power Bloom",
        "cobalt nickel supply DRC Indonesia",
        "GE Vernova grid orders backlog",
        "Eaton electrification cycle data center",
        "Schneider Electric EcoStruxure software",
    ]
    sources = [
        ("Lithium spot price stabilizes", "https://www.example.com/e/li-price", "Lithium carbonate prices stabilized after two-year drawdown; high-cost producers idled supply.", ["raw_inputs"], ["ALB", "SQM"]),
        ("Copper smelter capacity stays tight", "https://www.example.com/e/cu-smelter", "Smelter treatment charges hit multi-year lows, signaling structural copper concentrate shortage.", ["raw_inputs"], ["FCX"]),
        ("Albemarle cost-cuts deepen", "https://www.example.com/e/alb", "Albemarle announced further capex cuts and Australian operation curtailments.", ["raw_inputs"], ["ALB"]),
        ("CATL EV cell capacity grows 30%", "https://www.example.com/e/catl", "CATL's energy storage capacity guided up 30%, outside of EV cell business.", ["integrators"], ["CATL"]),
        ("LG Energy Solution wins US storage deal", "https://www.example.com/e/lge", "LG Energy signed a multi-GW storage cell supply deal with US developer.", ["integrators"], ["LG Energy"]),
        ("QuantumScape lab scale-up", "https://www.example.com/e/qs", "QuantumScape disclosed multi-layer cell production at pilot line.", ["enabling"], ["QS"]),
        ("First Solar Series 7 fully sold out", "https://www.example.com/e/fslr", "First Solar's Series 7 modules are sold out through 2026 at fixed pricing.", ["enabling"], ["FSLR"]),
        ("Enphase IQ8 US residential share", "https://www.example.com/e/enph", "Enphase's IQ8 microinverter holds dominant US residential share despite TPV pressure.", ["enabling"], ["ENPH"]),
        ("Quanta backlog grew 25%", "https://www.example.com/e/pwr", "Quanta Services reported transmission and renewables backlog up 25% YoY.", ["enabling", "integrators"], ["PWR"]),
        ("Eaton data center electrical growth", "https://www.example.com/e/etn", "Eaton's electrical segment, anchored by data centers, drove company growth.", ["enabling"], ["ETN"]),
        ("Schneider Electric EcoStruxure ARR", "https://www.example.com/e/schneider", "Schneider's software ARR grew 30% on data-center and grid-management subscriptions.", ["enabling"], ["Schneider"]),
        ("Vestas onshore order intake recovers", "https://www.example.com/e/vws", "Vestas reported sharply improved onshore order pricing.", ["integrators"], ["Vestas"]),
        ("Siemens Energy grid orders surge", "https://www.example.com/e/siemens", "Siemens Energy's grid technologies backlog hit a record on transmission upgrades.", ["enabling"], ["Siemens"]),
        ("NuScale postpones first SMR project", "https://www.example.com/e/smr", "NuScale's first SMR site delayed; sector-wide signal that SMR timeline slips.", ["integrators"], ["SMR"]),
        ("Holtec restarts Palisades nuclear", "https://www.example.com/e/holtec", "Holtec progressing on Palisades restart and SMR-300 deployment plans.", ["integrators"], ["Holtec"]),
        ("Oklo signs data-center PPAs", "https://www.example.com/e/oklo", "Oklo signed term sheets with multiple data-center developers for behind-the-meter SMRs.", ["integrators", "applications"], ["OKLO"]),
        ("Plug Power liquidity stretched", "https://www.example.com/e/plug", "Plug Power refinanced and disclosed reduced electrolyzer guidance.", ["enabling"], ["PLUG"]),
        ("Bloom data-center fuel cell wins", "https://www.example.com/e/be", "Bloom Energy disclosed fuel-cell deployments at AEP-served data centers.", ["enabling", "applications"], ["BE"]),
        ("Carrier heat pump growth slows", "https://www.example.com/e/carr", "Carrier guided to mid-single-digit residential heat pump growth, below prior outlook.", ["applications"], ["CARR"]),
        ("Trane data center HVAC backlog", "https://www.example.com/e/tt", "Trane Technologies cited data-center HVAC as fastest-growing applied vertical.", ["applications", "enabling"], ["TT"]),
        ("ChargePoint cash burn improves", "https://www.example.com/e/chpt", "ChargePoint narrowed losses on subscription mix shift.", ["applications"], ["CHPT"]),
        ("EVgo utilization breaks 25%", "https://www.example.com/e/evgo", "EVgo's network utilization passed 25% as fleet customers grew.", ["applications"], ["EVGO"]),
        ("GE Vernova grid backlog", "https://www.example.com/e/gev", "GE Vernova reported a record grid orders book; HVDC programs leading.", ["enabling"], ["GEV"]),
        ("Hubbell volume up on transformer demand", "https://www.example.com/e/hubb", "Hubbell's transmission and distribution segment growth led by data-center transformer demand.", ["enabling"], ["HUBB"]),
        ("Cobalt market remains oversupplied", "https://www.example.com/e/co", "DRC cobalt oversupply weighs on prices; downstream cell chemistries shifting LFP.", ["raw_inputs"], ["DRC"]),
        ("Nickel Indonesia oversupply", "https://www.example.com/e/ni", "Indonesian Class 1 nickel oversupply weighs on prices; LFP cell chemistries reduce nickel exposure.", ["raw_inputs"], ["Indonesia"]),
    ]
    nodes = [
        {"name": "Critical minerals (Cu, Li, Ni, Co)", "layer": "raw_inputs", "description": "Copper supply is tight; lithium has stabilized; nickel/cobalt remain oversupplied.", "public_names": ["FCX", "ALB", "SQM"], "private_names": [], "catalysts": ["Smelter TC/RC at multi-year lows"], "risks": ["LFP chemistry reduces nickel/cobalt need"], "sources": ["https://www.example.com/e/cu-smelter", "https://www.example.com/e/li-price", "https://www.example.com/e/alb", "https://www.example.com/e/co"], "confidence": 0.72},
        {"name": "Battery cells", "layer": "integrators", "description": "CATL and LG Energy Solution dominate; LFP for stationary storage growing fastest.", "public_names": [], "private_names": ["CATL", "LG Energy Solution", "Panasonic"], "catalysts": ["Energy storage capacity ramp"], "risks": ["Chinese overcapacity caps pricing"], "sources": ["https://www.example.com/e/catl", "https://www.example.com/e/lge"], "confidence": 0.7},
        {"name": "Solid-state batteries", "layer": "enabling", "description": "QuantumScape and Sila are the public-equity proxies; commercialization risk remains high.", "public_names": ["QS"], "private_names": ["Sila"], "catalysts": ["Pilot line cell shipments"], "risks": ["Timeline slippage"], "sources": ["https://www.example.com/e/qs"], "confidence": 0.4},
        {"name": "Solar PV", "layer": "enabling", "description": "First Solar sold out at fixed pricing; Enphase holds US residential.", "public_names": ["FSLR", "ENPH"], "private_names": [], "catalysts": ["Series 7 capacity ramp"], "risks": ["Tariff regime changes"], "sources": ["https://www.example.com/e/fslr", "https://www.example.com/e/enph"], "confidence": 0.74},
        {"name": "Wind", "layer": "integrators", "description": "Vestas pricing recovers; offshore segment still under stress.", "public_names": [], "private_names": ["Vestas"], "catalysts": ["Onshore pricing inflection"], "risks": ["Permitting delays"], "sources": ["https://www.example.com/e/vws"], "confidence": 0.6},
        {"name": "Grid build-out", "layer": "enabling", "description": "Quanta, GE Vernova, Eaton, Hubbell, Siemens Energy — the deepest moat in energy transition today.", "public_names": ["PWR", "GEV", "ETN", "HUBB"], "private_names": ["Siemens Energy", "Schneider Electric"], "catalysts": ["Record HVDC and transformer orders"], "risks": ["Customer concentration in hyperscalers"], "sources": ["https://www.example.com/e/pwr", "https://www.example.com/e/gev", "https://www.example.com/e/etn", "https://www.example.com/e/hubb", "https://www.example.com/e/siemens"], "confidence": 0.88},
        {"name": "Power electronics & software", "layer": "enabling", "description": "Schneider EcoStruxure and grid-orchestration software is the highest-margin layer.", "public_names": [], "private_names": ["Schneider Electric"], "catalysts": ["Software ARR growth"], "risks": ["Hyperscaler in-house buildouts"], "sources": ["https://www.example.com/e/schneider"], "confidence": 0.66},
        {"name": "Nuclear (SMRs, restarts)", "layer": "integrators", "description": "Holtec restarts, Oklo behind-the-meter SMRs for data centers — narrative-rich, lumpy execution.", "public_names": ["SMR", "OKLO"], "private_names": ["Holtec"], "catalysts": ["Data-center PPA signings"], "risks": ["Project delays (NuScale precedent)"], "sources": ["https://www.example.com/e/smr", "https://www.example.com/e/holtec", "https://www.example.com/e/oklo"], "confidence": 0.5},
        {"name": "Hydrogen", "layer": "enabling", "description": "Plug Power and Bloom remain narrative plays; Bloom has near-term data-center pull.", "public_names": ["PLUG", "BE"], "private_names": [], "catalysts": ["Behind-the-meter data-center fuel cells"], "risks": ["Tax-credit dependency"], "sources": ["https://www.example.com/e/plug", "https://www.example.com/e/be"], "confidence": 0.42},
        {"name": "Electrification end-markets", "layer": "applications", "description": "Heat pumps (Carrier, Trane), EV charging (ChargePoint, EVgo).", "public_names": ["CARR", "TT", "CHPT", "EVGO"], "private_names": [], "catalysts": ["Trane data-center HVAC growth"], "risks": ["Residential heat-pump growth slowing"], "sources": ["https://www.example.com/e/carr", "https://www.example.com/e/tt", "https://www.example.com/e/chpt", "https://www.example.com/e/evgo"], "confidence": 0.55},
    ]
    picks = [
        "ETN — best risk/reward in grid electrification: data-center + reshoring tailwinds, durable margins.",
        "PWR — Quanta is the prime contractor for the grid build; backlog visibility through decade.",
        "FSLR — sold-out Series 7 at fixed prices = pricing power that doesn't show up in spot data.",
        "GEV — GE Vernova's HVDC franchise is uniquely positioned for inter-regional transmission.",
        "FCX — leveraged exposure to structurally tight copper supply, with a buyback optionality.",
    ]
    headlines = [
        "Most 'energy transition' returns the next 3 years come from the grid, not the cells.",
        "Solid-state batteries and green hydrogen remain prove-it stories — public equity is too early.",
        "Hyperscaler power demand is the dominant marginal buyer; data center is the new utility customer.",
        "Chinese cell overcapacity caps battery margins; the picks-and-shovels (grid, power electronics) escape.",
        "Nuclear-SMR euphoria forgets the NuScale precedent — first-of-kind timelines slip.",
    ]
    summary = (
        "The most durable energy-transition exposure for the next three years is the grid build, not the "
        "renewable generation cell itself. Quanta, Eaton, GE Vernova, Hubbell, and Schneider are the "
        "picks-and-shovels for hyperscaler-driven and reshoring-driven electrical-infrastructure capex. "
        "Renewable cell economics (solar modules, lithium batteries) are pressured by Chinese overcapacity. "
        "Solid-state batteries, hydrogen, and SMRs are narrative-rich but execution-poor; public equity is "
        "the wrong vehicle for those bets today. Copper supply is the most structural commodity tailwind."
    )
    diagram = """graph LR
  subgraph raw[Raw inputs]
    r1["Critical minerals<br/>FCX, ALB"]
  end
  subgraph enabling[Enabling]
    e1["Grid build-out<br/>PWR, GEV, ETN, HUBB"]
    e2["Power electronics SW<br/>Schneider"]
    e3["Solar PV<br/>FSLR, ENPH"]
    e4["Solid-state<br/>QS"]
    e5["Hydrogen<br/>PLUG, BE"]
  end
  subgraph integrators[Integrators]
    i1["Battery cells<br/>CATL, LGES"]
    i2["Wind<br/>Vestas"]
    i3["Nuclear / SMR<br/>SMR, OKLO, Holtec"]
  end
  subgraph applications[Applications]
    a1["Electrification<br/>CARR, TT"]
    a2["EV charging<br/>CHPT, EVGO"]
  end
  r1 --> i1
  r1 --> e1
  e3 --> i2
  i1 --> a2
  e1 --> a1
  i3 --> a1"""
    return _ThemePack("Energy transition", queries, sources, nodes, picks, headlines, summary, diagram)


def _generic_pack(theme: str) -> _ThemePack:
    """Plausible fallback for any theme outside our pre-canned set."""
    queries = [
        f"{theme} raw material supply chain",
        f"{theme} key suppliers and manufacturers",
        f"{theme} market leaders public companies",
        f"{theme} bottlenecks and chokepoints",
        f"{theme} end market applications",
        f"{theme} private companies funding",
        f"{theme} regulatory landscape",
        f"{theme} geopolitical risks",
        f"{theme} capital expenditure trends",
        f"{theme} competitive moats",
        f"{theme} technology substitution risks",
        f"{theme} pricing trends",
        f"{theme} growth catalysts 2026",
        f"{theme} historical analog cycles",
        f"{theme} consensus sell-side view",
    ]
    sources = [
        (f"{theme}: market structure overview", f"https://www.example.com/g/{i}", f"Demo source #{i} describing one facet of the {theme} value chain.", ["raw_inputs" if i < 4 else "enabling" if i < 10 else "integrators" if i < 15 else "applications"], ["DEMO"])
        for i in range(1, 21)
    ]
    nodes = [
        {"name": f"Raw inputs for {theme}", "layer": "raw_inputs", "description": "Demo layer — replace with real data by setting API keys.", "public_names": ["DEMO"], "private_names": [], "catalysts": ["Demo catalyst"], "risks": ["Demo risk"], "sources": ["https://www.example.com/g/1"], "confidence": 0.5},
        {"name": f"Enabling technology in {theme}", "layer": "enabling", "description": "Demo layer.", "public_names": ["DEMO"], "private_names": ["DemoCo"], "catalysts": ["Demo catalyst"], "risks": ["Demo risk"], "sources": ["https://www.example.com/g/5"], "confidence": 0.5},
        {"name": f"{theme} integrators", "layer": "integrators", "description": "Demo layer.", "public_names": ["DEMO"], "private_names": [], "catalysts": ["Demo catalyst"], "risks": ["Demo risk"], "sources": ["https://www.example.com/g/10"], "confidence": 0.5},
        {"name": f"{theme} end-market applications", "layer": "applications", "description": "Demo layer.", "public_names": ["DEMO"], "private_names": [], "catalysts": ["Demo catalyst"], "risks": ["Demo risk"], "sources": ["https://www.example.com/g/15"], "confidence": 0.5},
    ]
    picks = [
        f"DEMO — demo pick for {theme}; configure API keys for real output.",
        "DEMO — second demo pick.",
        "DEMO — third demo pick.",
        "DEMO — fourth demo pick.",
        "DEMO — fifth demo pick.",
    ]
    headlines = [
        f"Demo mode does not produce real analysis for {theme}.",
        "Set GROQ_API_KEY + ANTHROPIC_API_KEY and disable demo mode for real output.",
        "The graph below shows the canonical raw → enabling → integrators → applications template.",
    ]
    summary = (
        f"This is demo output for the theme '{theme}'. The full graph exercised the real researcher, "
        "skeptic, and synthesizer agents — only the LLM calls and search results were stubbed. "
        "To produce a real thesis, set API keys and run without the --demo flag."
    )
    diagram = f"""graph LR
  subgraph raw[Raw inputs]
    r1["Raw inputs<br/>DEMO"]
  end
  subgraph enabling[Enabling]
    e1["Enabling<br/>DEMO"]
  end
  subgraph integrators[Integrators]
    i1["Integrators<br/>DEMO"]
  end
  subgraph applications[Applications]
    a1["Applications<br/>DEMO"]
  end
  r1 --> e1 --> i1 --> a1"""
    return _ThemePack(theme, queries, sources, nodes, picks, headlines, summary, diagram)


_PACKS: dict[str, _ThemePack] = {}


def _pack_for(theme: str) -> _ThemePack:
    if not _PACKS:
        for p in (_ai_infra_pack(), _robotics_pack(), _energy_pack()):
            _PACKS[_normalize(p.theme)] = p
    return _PACKS.get(_normalize(theme)) or _generic_pack(theme)


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# --- Fake search ------------------------------------------------------------


class DemoSearch:
    """Plays back canned search results from the matching theme pack."""

    def __init__(self, theme: str, trace: TraceWriter | None = None) -> None:
        self._pack = _pack_for(theme)
        self._trace = trace
        self._cursor = 0

    def search(self, query: str, k: int = 8) -> list[SearchResult]:
        time.sleep(random.uniform(0.05, 0.15))
        results: list[SearchResult] = []
        for _ in range(min(k, len(self._pack.sources))):
            title, url, snippet, _layers, _ents = self._pack.sources[
                self._cursor % len(self._pack.sources)
            ]
            self._cursor += 1
            results.append(
                SearchResult(title=title, url=url, snippet=snippet, provider="demo")
            )
        if self._trace is not None:
            self._trace.write(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "kind": "search",
                    "note": f"q={query!r} n={len(results)} provider=demo",
                }
            )
        return results


def demo_fetch_source(url: str, cache: Any, trace: TraceWriter | None = None, **_kw: Any) -> str:
    """Drop-in for ``tools.fetch_source``; returns a snippet keyed by URL."""
    for pack in (_ai_infra_pack(), _robotics_pack(), _energy_pack()):
        for title, u, snippet, _, _ in pack.sources:
            if u == url:
                if trace is not None:
                    trace.write(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "kind": "fetch",
                            "note": f"url={url} demo",
                        }
                    )
                return f"{title}. {snippet}"
    return "Demo source content."


# --- Fake LLM ---------------------------------------------------------------


@dataclass
class _Counters:
    rounds: int = 0
    nodes_emitted: int = 0
    critiques_emitted: int = 0
    summaries_emitted: int = 0


class DemoLLM(LLMClient):
    """Drop-in replacement for ``LLMClient`` in demo mode.

    Routes structured-output requests to canned data from the matching
    theme pack. Records realistic-looking trace events so the metrics
    panel shows useful numbers.
    """

    # Per-call fake usage. Picked so the cheap-tier share lands around 85%.
    _USAGE = {
        "small":     {"prompt": 600,  "completion": 250,  "cost": 0.00015},
        "reasoning": {"prompt": 1200, "completion": 600,  "cost": 0.0015},
        "quality":   {"prompt": 4000, "completion": 1500, "cost": 0.02},
    }

    def __init__(
        self,
        theme: str,
        trace: TraceWriter | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        # Skip super().__init__ — we don't need the real config or LiteLLM.
        self.trace = trace or TraceWriter()
        self.max_cost_usd = max_cost_usd
        self._theme = theme
        self._pack = _pack_for(theme)
        self._budgets = {
            "researcher_context_tokens": 60_000,
            "skeptic_context_tokens": 30_000,
            "synthesizer_context_tokens": 80_000,
        }
        self._instructor_cfg = {"max_retries": 0}
        self._counters = _Counters()

        @dataclass
        class _T:
            primary: str = "demo/stub"
            fallbacks: list[str] = field(default_factory=list)
            max_tokens: int = 1024
            temperature: float = 0.3
            timeout_s: int = 30
            enable_prompt_cache: bool = False

        self._tiers = {"small": _T(), "reasoning": _T(), "quality": _T(enable_prompt_cache=True)}

    @property
    def budgets(self) -> dict[str, int]:
        return self._budgets

    def tier(self, name: str) -> Any:
        return self._tiers[name]

    def complete(
        self,
        tier: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
        **_: Any,
    ) -> Any:
        usage = self._USAGE[tier]
        delay = {"small": 0.08, "reasoning": 0.25, "quality": 0.7}[tier]
        time.sleep(delay + random.uniform(-0.03, 0.07))
        self.trace.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": "llm",
                "tier": tier,
                "model": f"demo/{tier}",
                "prompt_tokens": usage["prompt"],
                "completion_tokens": usage["completion"],
                "cost_usd": usage["cost"],
                "latency_ms": int(delay * 1000),
                "note": f"demo structured={response_model is not None}",
            }
        )
        if response_model is None:
            return "demo response"
        return self._materialize(response_model, messages)

    # ---- dispatch ----------------------------------------------------------

    def _materialize(
        self, response_model: type[BaseModel], messages: list[dict[str, Any]]
    ) -> BaseModel:
        name = response_model.__name__
        if name == "QueryPlan":
            return self._query_plan(messages)
        if name == "SourceSummary":
            return self._source_summary(messages)
        if name == "CoverageReport":
            return self._coverage_report()
        if name == "NodeList":
            return self._node_list()
        if name == "Critique":
            return self._critique(messages)
        if name == "FinalOutput":
            return self._final_output()
        if name == "_Headlines" or "Headlines" in name:
            # Skeptic rollup helper class.
            return response_model.model_validate({"headlines": self._pack.headlines})
        # Eval judge fallback.
        if name == "JudgeScores":
            from evals.judges import DimensionScore, JudgeScores  # type: ignore

            return JudgeScores(
                coverage=DimensionScore(score=4, justification="demo"),
                specificity=DimensionScore(score=4, justification="demo"),
                novelty=DimensionScore(score=3, justification="demo"),
                actionability=DimensionScore(score=4, justification="demo"),
            )
        # Unknown structured request — let Instructor's validator handle it
        # the same way the real client would (raise).
        raise NotImplementedError(f"DemoLLM has no canned response for {name}")

    # ---- canned responses --------------------------------------------------

    def _query_plan(self, messages: list[dict[str, Any]]) -> QueryPlan:
        # Pull a fresh slice each call so multiple rounds look different.
        start = self._counters.rounds * 6
        slice_ = self._pack.queries[start : start + 6] or self._pack.queries[:6]
        self._counters.rounds += 1
        layers = ["raw_inputs", "enabling", "integrators", "applications"]
        return QueryPlan(
            queries=[
                SearchQuery(query=q, target_layer=layers[i % 4], rationale="demo")
                for i, q in enumerate(slice_)
            ]
        )

    def _source_summary(self, messages: list[dict[str, Any]]) -> SourceSummary:
        # Extract URL/title from the user message so the summary references
        # the same source the researcher fed in (keeps the URL set consistent).
        user = next(
            (m for m in messages if m.get("role") == "user"), {"content": ""}
        )
        text = str(user.get("content", ""))
        url = _extract(text, r"URL:\s*(\S+)")
        title = _extract(text, r"Title:\s*(.+)")
        # Look up the canonical source in the pack so layer_hints / entities
        # match what the synthesizer expects.
        for t, u, snippet, layers, ents in self._pack.sources:
            if u == url:
                self._counters.summaries_emitted += 1
                return SourceSummary(
                    url=u,
                    title=t,
                    summary=snippet,
                    layer_hints=layers,  # type: ignore[arg-type]
                    entities=ents,
                    relevance=round(random.uniform(0.55, 0.92), 2),
                )
        # Fallback: synthesize a plausible summary.
        self._counters.summaries_emitted += 1
        return SourceSummary(
            url=url or "https://www.example.com/g/x",
            title=title or "Demo source",
            summary="Demo summary derived from a source the demo pack did not pre-cache.",
            layer_hints=["enabling"],
            entities=["DEMO"],
            relevance=0.5,
        )

    def _coverage_report(self) -> CoverageReport:
        # First audit: not done; subsequent: done. Mirrors the researcher's loop.
        done = self._counters.rounds >= 2
        return CoverageReport(
            per_layer_counts={
                "raw_inputs": 3,
                "enabling": 6,
                "integrators": 4,
                "applications": 3,
            },
            gaps=[],
            done=done,
        )

    def _node_list(self) -> NodeList:
        nodes = [Node.model_validate(n) for n in self._pack.nodes]
        self._counters.nodes_emitted = len(nodes)
        return NodeList(nodes=nodes)

    def _critique(self, messages: list[dict[str, Any]]) -> Critique:
        user = next((m for m in messages if m.get("role") == "user"), {"content": ""})
        text = str(user.get("content", ""))
        node_name = _extract(text, r"name:\s*(.+)") or "Unknown node"
        angle = _extract(text, r"Attack angle:\s*(\S+)") or "bottleneck_vs_commodity"
        verdict = random.choices(["weak", "mixed", "strong"], weights=[1, 2, 3])[0]
        argument = {
            "bottleneck_vs_commodity": f"{node_name} screens as a chokepoint today, but the relevant question is whether pricing power persists once a second supplier qualifies.",
            "substitution_risk": f"{node_name}'s substitution path is real but underwritten — three years out is the right horizon, not one.",
            "priced_in": f"Consensus on {node_name} is constructive; the variant view is on duration, not direction.",
        }.get(angle, "Demo critique.")
        self._counters.critiques_emitted += 1
        return Critique(
            node_name=node_name,
            angle=angle,  # type: ignore[arg-type]
            verdict=verdict,  # type: ignore[arg-type]
            argument=argument,
            suggested_change="demote confidence by 0.1" if verdict == "weak" else "",
        )

    def _final_output(self) -> FinalOutput:
        nodes = [Node.model_validate(n) for n in self._pack.nodes]
        thesis = Thesis(
            theme=self._pack.theme,
            summary=self._pack.summary,
            nodes=nodes,
            investable_picks=self._pack.investable_picks,
            skeptic_notes=self._pack.headlines,
            generated_at=datetime.now(timezone.utc),
        )
        memo = _build_memo(self._pack, nodes)
        return FinalOutput(thesis=thesis, memo_md=memo, diagram_mmd=self._pack.diagram_mmd)


# --- helpers ----------------------------------------------------------------


def _extract(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(1).strip().splitlines()[0].strip()


def _build_memo(pack: _ThemePack, nodes: list[Node]) -> str:
    sections: list[str] = [
        f"# {pack.theme} — Value Chain Thesis",
        "",
        "## Summary",
        "",
        pack.summary,
        "",
        "## Value chain",
        "",
    ]
    for layer, label in (
        ("raw_inputs", "Raw inputs"),
        ("enabling", "Enabling layer"),
        ("integrators", "Integrators"),
        ("applications", "Applications"),
    ):
        layer_nodes = [n for n in nodes if n.layer == layer]
        if not layer_nodes:
            continue
        sections.append(f"### {label}")
        sections.append("")
        for n in layer_nodes:
            tickers = ", ".join(n.public_names) or "—"
            sections.append(f"- **{n.name}** ({tickers}) — {n.description}")
        sections.append("")
    sections.append("## Investable picks")
    sections.append("")
    for i, p in enumerate(pack.investable_picks, start=1):
        sections.append(f"{i}. {p}")
    sections.append("")
    sections.append("## What we'd have to be wrong about")
    sections.append("")
    for h in pack.headlines:
        sections.append(f"- {h}")
    sections.append("")
    sections.append("## Sources")
    sections.append("")
    urls = sorted({str(u) for n in nodes for u in n.sources})
    for u in urls:
        sections.append(f"- {u}")
    return "\n".join(sections)
