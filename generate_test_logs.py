"""
generate_test_logs.py
=====================
Generates 20 test files in artifacts/data/:
  - 10 machine constant files  (various formats: CSV, INI, JSON, YAML, XML, TSV)
  - 10 recipe detail files     (various formats: JSON, XML, CSV, YAML, TSV, TXT)

Designed to exercise the full DeepParse pipeline including:
  - LLM mask synthesis (novel tokens not in universal masks)
  - Adaptive feedback loop (low initial parse rate on some files)
  - All format loaders
  - Edge cases: scientific notation, nested YAML, multi-step XML recipes
"""

import json
import pathlib
import random
import textwrap

random.seed(99)
OUT = pathlib.Path("artifacts/data")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: write file with no BOM
# ---------------------------------------------------------------------------
def write(path: str, content: str) -> None:
    p = OUT / path
    p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    print(f"  Created: {p}")


# ===========================================================================
# MACHINE CONSTANT FILES (10)
# ===========================================================================

# 1 — CSV (key-value table style)
write("mc_01_nova_machine_constants.csv", """
ToolID,EQP_NOVA_001
Vendor,NovaStar Systems
ProcessType,Optical Metrology
SoftwareVersion,VER_NOVA_4.2.1
MaxWaferDiameter,300
MinPressure,1.50e-7
MaxTemperature,85.0
CalibrationDate,2026-01-15
SensorCount,12
NetworkAddress,NET_NOVA_001
LotCapacity,25
SlotCount,50
""")

# 2 — INI (sectioned constants)
write("mc_02_plasmacore_machine_constants.ini", """
[ToolIdentification]
ToolID = EQP_PC_001
Vendor = PlasmaCoreInc
SoftwareVersion = VER_PC_3.8.0
MachineType = MTYPE_CCP_001

[ProcessParameters]
MaxRFPower = 3000.0
MinPressure = 2.00e-6
MaxTemperature = 400.0
GasFlowMax = 500.0
BiasVoltage = -150.0
PlasmaFrequency = 13.56

[Calibration]
LastCalibDate = 2026-02-10
CalibOperator = CREATOR_PC_001
RFMatchOffset = 0.023
PressureOffset = -0.0015

[Network]
NetworkAddress = NET_PC_001
IPAddress = 192.168.1.101
Port = 5025
""")

# 3 — JSON
write("mc_03_chipconst_machine_constants.json", json.dumps({
    "ToolID": "EQP_CC_001",
    "Vendor": "ChipConst Technologies",
    "ProcessType": "LPCVD",
    "SoftwareVersion": "VER_CC_2.5.3",
    "Constants": {
        "MaxTemp":       {"value": 750.0,    "unit": "C",     "type": "float"},
        "MinPressure":   {"value": 5.00e-8,  "unit": "Torr",  "type": "float"},
        "GasFlow_SiH4":  {"value": 50.0,     "unit": "sccm",  "type": "float"},
        "GasFlow_N2":    {"value": 200.0,    "unit": "sccm",  "type": "float"},
        "RampRate":      {"value": 5.0,      "unit": "C/min", "type": "float"},
        "SoakTime":      {"value": 300,      "unit": "s",     "type": "int"},
        "WaferSpacing":  {"value": 4.76,     "unit": "mm",    "type": "float"},
    },
    "Sensors": [
        {"SensorID": "SENSOR_0001", "Type": "Thermocouple", "Location": "Top"},
        {"SensorID": "SENSOR_0002", "Type": "Thermocouple", "Location": "Middle"},
        {"SensorID": "SENSOR_0003", "Type": "Pressure",     "Location": "Chamber"},
    ],
    "LastModified": "2026-02-18T08:00:00Z",
    "ApprovedBy": "CREATOR_CC_001"
}, indent=2))

# 4 — YAML
write("mc_04_thermaldyn_machine_constants.yaml", """
ToolID: EQP_TD_001
Vendor: ThermalDyn Systems
ProcessType: Thermal Oxidation
SoftwareVersion: VER_TD_3.9.1

ProcessParameters:
  MaxTemperature: 1100.0
  MinTemperature: 800.0
  RampRateUp: 10.0
  RampRateDown: 5.0
  O2FlowMax: 10.0
  H2FlowMax: 5.0
  N2FlowMax: 20.0
  PressureSetpoint: 760.0

Zones:
  - ZoneID: ZONE_TD_001
    Position: Top
    TempOffset: 2.5
  - ZoneID: ZONE_TD_002
    Position: Middle
    TempOffset: 0.0
  - ZoneID: ZONE_TD_003
    Position: Bottom
    TempOffset: -1.8

Calibration:
  LastCalibDate: 2026-01-20
  Operator: CREATOR_TD_001
  ThermocoupleOffset: 0.5
""")

# 5 — XML
write("mc_05_ionbeam_machine_constants.xml", """<?xml version="1.0" encoding="UTF-8"?>
<MachineConstants ToolID="EQP_IB_001" Version="VER_IB_5.1.0">
  <Identification>
    <Vendor>IonBeam Technologies</Vendor>
    <ProcessType>Ion Implantation</ProcessType>
    <SerialNumber>SN_IB_20240315</SerialNumber>
  </Identification>
  <Constants>
    <Constant name="BeamEnergy"    value="80.0"    unit="keV"  type="float"/>
    <Constant name="BeamCurrent"   value="2.5"     unit="mA"   type="float"/>
    <Constant name="DoseTarget"    value="1.00e15" unit="cm-2" type="float"/>
    <Constant name="ScanFrequency" value="1000"    unit="Hz"   type="int"/>
    <Constant name="TiltAngle"     value="7.0"     unit="deg"  type="float"/>
    <Constant name="TwistAngle"    value="22.0"    unit="deg"  type="float"/>
    <Constant name="ChamberPressure" value="5.00e-7" unit="Torr" type="float"/>
  </Constants>
  <Sensors>
    <Sensor id="SENSOR_0001" type="Faraday" location="EndStation"/>
    <Sensor id="SENSOR_0002" type="Pressure" location="Analyzer"/>
    <Sensor id="SENSOR_0003" type="Pressure" location="EndStation"/>
  </Sensors>
  <Calibration lastDate="2026-02-01" operator="CREATOR_IB_001"/>
</MachineConstants>
""")

# 6 — TSV
write("mc_06_sputterpro_machine_constants.tsv",
"Parameter\tValue\tUnit\tType\n" +
"\n".join([
    "ToolID\tEQP_SP_001\t-\tstring",
    "Vendor\tSputter Pro Systems\t-\tstring",
    "ProcessType\tPVD Sputtering\t-\tstring",
    "SoftwareVersion\tVER_SP_2.8.0\t-\tstring",
    "MaxPower\t20000.0\tW\tfloat",
    "ArgonFlow\t50.0\tsccm\tfloat",
    "BasePressure\t1.50e-8\tTorr\tfloat",
    "ProcessPressure\t3.00e-3\tTorr\tfloat",
    "SubstrateTemp\t200.0\tC\tfloat",
    "TargetVoltage\t-450.0\tV\tfloat",
    "DepositionRate\t2.5\tnm/min\tfloat",
    "NetworkAddress\tNET_SP_001\t-\tstring",
    "LastCalibDate\t2026-01-28\t-\tdate",
]))

# 7 — TXT (syslog-style machine status)
write("mc_07_cleantech_machine_constants.txt", """
Machine:MCH_CT_001 ToolID=EQP_CT_001 Vendor=CleanTech Status=IDLE
Machine:MCH_CT_001 SoftwareVersion=VER_CT_1.4.2 NetworkAddress=NET_CT_001
Machine:MCH_CT_001 Constant MaxSpinSpeed=3000.0 rpm Limit=4500.0
Machine:MCH_CT_001 Constant DIWaterFlow=2.5 lpm Limit=5.0
Machine:MCH_CT_001 Constant ChemicalFlow=0.5 lpm Limit=1.0
Machine:MCH_CT_001 Constant SpinTime=60 s Limit=300
Machine:MCH_CT_001 Constant DryTemp=80.0 C Limit=120.0
Machine:MCH_CT_001 Sensor SENSOR_0001 Type=FlowMeter Location=DIWater Reading=2.48 lpm
Machine:MCH_CT_001 Sensor SENSOR_0002 Type=Pressure Location=N2 Reading=1.02 bar
Machine:MCH_CT_001 Sensor SENSOR_0003 Type=Temperature Location=Chuck Reading=25.3 C
Machine:MCH_CT_001 Calibration LastDate=2026-02-05 Operator=CREATOR_CT_001
Machine:MCH_CT_001 Network IP=192.168.1.105 Port=8080 Protocol=SECS2
""")

# 8 — CSV (multi-column structured)
write("mc_08_metrovision_machine_constants.csv", """
ParameterName,Value,Unit,MinLimit,MaxLimit,Alarm
ToolID,EQP_MV_001,,,, 
SoftwareVersion,VER_MV_6.3.0,,,,
LaserWavelength,632.8,nm,630.0,636.0,False
SpotSize,2.5,um,2.0,3.0,False
ScanSpeed,10.0,mm/s,1.0,50.0,False
MeasurementRange,500.0,nm,0.0,1000.0,False
Repeatability,0.1,nm,,,False
SensorID,SENSOR_0001,,,,
NetworkAddress,NET_MV_001,,,,
CalibrationDate,2026-02-12,,,,
CalibrationOffset,0.015,nm,,,
""")

# 9 — YAML (nested with multiple modules)
write("mc_09_annealtech_machine_constants.yaml", """
ToolID: EQP_AT_001
Vendor: AnnealTech Corp
ProcessType: RTP Anneal
SoftwareVersion: VER_AT_4.0.2

Modules:
  - ModuleID: MOD_LAMP_001
    Type: HalogenLamp
    PowerRating: 50000.0
    WavelengthRange: 0.4-4.0
  - ModuleID: MOD_PYROMETER_001
    Type: Pyrometer
    Range: 400.0-1200.0
    Emissivity: 0.65
  - ModuleID: MOD_GAS_001
    Type: GasPanel
    Gases: [N2, O2, NH3]

ProcessParameters:
  MaxTemp: 1050.0
  RampRateMax: 150.0
  SoakTimeMax: 300
  N2FlowMax: 20.0
  O2FlowMax: 5.0

Sensors:
  - SensorID: SENSOR_0001
    Type: Pyrometer
    SampleRate: 100
  - SensorID: SENSOR_0002
    Type: Thermocouple
    Location: Edge

NetworkConfig:
  NetworkAddress: NET_AT_001
  IPAddress: 192.168.1.108
  Protocol: HSMS
""")

# 10 — INI (complex multi-section)
write("mc_10_polishmaster_machine_constants.ini", """
[ToolIdentification]
ToolID = EQP_PM_001
Vendor = PolishMaster Inc
ProcessType = CMP
SoftwareVersion = VER_PM_3.2.1
SerialNumber = SN_PM_20230601

[PlatenParameters]
Platen1Speed = 93.0
Platen2Speed = 87.0
Platen3Speed = 0.0
CarrierSpeed = 87.0
Oscillation = 20.0
OscillationFreq = 0.5

[PressureParameters]
DownForce = 3.5
RetainingRing = 4.2
InnerTube = 2.8
OuterTube = 3.1
EdgeTube = 2.5

[SlurryParameters]
SlurryFlow = 200.0
DIWaterFlow = 300.0
SlurryTemp = 22.0
ConcentrationTarget = 12.5

[Sensors]
Sensor1 = SENSOR_0001
Sensor2 = SENSOR_0002
Sensor3 = SENSOR_0003
Sensor4 = SENSOR_0004

[Network]
NetworkAddress = NET_PM_001
IPAddress = 192.168.1.110
""")


# ===========================================================================
# RECIPE DETAIL FILES (10)
# ===========================================================================

# 1 — JSON (multi-step etch recipe)
write("rd_01_plasmacore_recipe_details.json", json.dumps({
    "RecipeID": "RCP_PC_ETCH_001",
    "ToolID": "EQP_PC_001",
    "LotID": "LOT_PC_001",
    "ProcessType": "Dry Etch",
    "CreatedBy": "CREATOR_PC_001",
    "CreatedAt": "2026-02-18T08:00:00Z",
    "Steps": [
        {"StepID": "PURGE",     "Time": 30,    "Pressure": 1.0e-5, "RF": 0,      "Gas": "N2",  "Flow": 200},
        {"StepID": "STRIKE",    "Time": 10,    "Pressure": 5.0e-3, "RF": 100,    "Gas": "Ar",  "Flow": 50},
        {"StepID": "MAIN_ETCH", "Time": 120,   "Pressure": 5.0e-3, "RF": 500,    "Gas": "CF4", "Flow": 50},
        {"StepID": "OVERETCH",  "Time": 30,    "Pressure": 5.0e-3, "RF": 300,    "Gas": "CF4", "Flow": 30},
        {"StepID": "PURGE2",    "Time": 20,    "Pressure": 1.0e-5, "RF": 0,      "Gas": "N2",  "Flow": 200},
    ],
    "TargetDepth": 450.0,
    "Selectivity": 20.0,
    "ExpirationDate": "2027-02-18"
}, indent=2))

# 2 — XML (lithography recipe)
write("rd_02_nova_recipe_details.xml", """<?xml version="1.0" encoding="UTF-8"?>
<Recipe RecipeID="RCP_NOVA_LITHO_001" ToolID="EQP_NOVA_001" Version="VER_NOVA_4.2.1">
  <Header>
    <LotID>LOT_NOVA_001</LotID>
    <WaferID>WFR_NOVA_001</WaferID>
    <CreatedBy>CREATOR_NOVA_001</CreatedBy>
    <CreatedAt>2026-02-18T09:00:00Z</CreatedAt>
    <ExpirationDate>2027-06-30</ExpirationDate>
  </Header>
  <IlluminationSettings>
    <Na>0.85</Na>
    <Inner>0.75</Inner>
    <Outer>0.95</Outer>
    <WavelengthNm>193.0</WavelengthNm>
    <PulseEnergy>1.50e-3</PulseEnergy>
  </IlluminationSettings>
  <Steps>
    <Step number="1" name="ALIGN"   duration="15.0" focusOffset="0.0"/>
    <Step number="2" name="EXPOSE"  duration="0.5"  dose="35.0" energy="1.50e-3"/>
    <Step number="3" name="MEASURE" duration="5.0"  overlay="true"/>
  </Steps>
  <Reticle ReticleID="RTL_0001" DOE="DOE_0001"/>
  <QualityTargets CDTarget="45.0" OverlayTarget="2.5" FocusTarget="0.0"/>
</Recipe>
""")

# 3 — CSV (PVD recipe)
write("rd_03_sputterpro_recipe_details.csv", """
Parameter,Value,Unit,StepID
RecipeID,RCP_SP_PVD_001,,HEADER
ToolID,EQP_SP_001,,HEADER
LotID,LOT_SP_001,,HEADER
CreatedBy,CREATOR_SP_001,,HEADER
ExpirationDate,2027-03-31,,HEADER
BaselinePressure,1.50e-8,Torr,PUMP
ArgonFlow_Step1,20.0,sccm,PRESPUT
Power_Step1,1000.0,W,PRESPUT
Time_Step1,60,s,PRESPUT
ArgonFlow_Step2,50.0,sccm,DEPOSITION
Power_Step2,5000.0,W,DEPOSITION
Time_Step2,300,s,DEPOSITION
Thickness_Target,500.0,nm,DEPOSITION
DepositionRate,1.67,nm/s,DEPOSITION
SubstrateTemp,200.0,C,DEPOSITION
ArgonFlow_Step3,10.0,sccm,COOLDOWN
Power_Step3,0.0,W,COOLDOWN
Time_Step3,120,s,COOLDOWN
""")

# 4 — YAML (CVD recipe with nested steps)
write("rd_04_chipconst_recipe_details.yaml", """
RecipeID: RCP_CC_CVD_001
ToolID: EQP_CC_001
LotID: LOT_CC_001
ProcessType: LPCVD SiN
CreatedBy: CREATOR_CC_001
ExpirationDate: 2027-04-15

Steps:
  - StepID: PUMP_DOWN
    Duration: 300
    TargetPressure: 1.00e-6
    Temperature: 300.0
    GasFlows: []

  - StepID: RAMP_TEMP
    Duration: 600
    TargetPressure: 1.00e-6
    Temperature: 750.0
    RampRate: 5.0
    GasFlows:
      - Gas: N2
        Flow: 100.0

  - StepID: DEPOSITION
    Duration: 1800
    TargetPressure: 2.50e-1
    Temperature: 750.0
    GasFlows:
      - Gas: SiH4
        Flow: 50.0
      - Gas: NH3
        Flow: 150.0
      - Gas: N2
        Flow: 200.0

  - StepID: PURGE
    Duration: 120
    TargetPressure: 1.00e-6
    Temperature: 600.0
    GasFlows:
      - Gas: N2
        Flow: 500.0

Targets:
  ThicknessNm: 100.0
  RefractiveIndex: 2.0
  StressMP: -200.0
""")

# 5 — TXT (syslog-style recipe execution log)
write("rd_05_ionbeam_recipe_details.txt", """
2026-02-18T10:00:00Z RecipeStart RecipeID=RCP_IB_IMP_001 ToolID=EQP_IB_001 LotID=LOT_IB_001
2026-02-18T10:00:01Z StepStart StepID=SETUP BeamEnergy=80.0keV Current=2.5mA Tilt=7.0deg
2026-02-18T10:00:05Z StepStart StepID=ALIGN WaferID=WFR_IB_001 SlotID=SLOT_001
2026-02-18T10:00:15Z StepStart StepID=IMPLANT DoseTarget=1.00e15 ScanSpeed=500.0mm/s
2026-02-18T10:02:35Z DoseComplete WaferID=WFR_IB_001 DoseAchieved=1.001e15 Uniformity=0.8pct
2026-02-18T10:02:36Z StepStart StepID=IMPLANT WaferID=WFR_IB_002 SlotID=SLOT_002
2026-02-18T10:04:58Z DoseComplete WaferID=WFR_IB_002 DoseAchieved=9.998e14 Uniformity=0.9pct
2026-02-18T10:04:59Z StepStart StepID=IMPLANT WaferID=WFR_IB_003 SlotID=SLOT_003
2026-02-18T10:07:22Z DoseComplete WaferID=WFR_IB_003 DoseAchieved=1.002e15 Uniformity=0.7pct
2026-02-18T10:07:23Z RecipeEnd RecipeID=RCP_IB_IMP_001 WafersProcessed=3 TotalTime=443s Status=COMPLETE
2026-02-18T10:07:24Z MetrologyRequest LotID=LOT_IB_001 RecipeID=RCP_IB_IMP_001 ToolID=EQP_MV_001
""")

# 6 — TSV (CMP recipe)
write("rd_06_polishmaster_recipe_details.tsv",
"StepID\tParameter\tValue\tUnit\tDuration\n" +
"\n".join([
    "HEADER\tRecipeID\tRCP_PM_CMP_001\t-\t0",
    "HEADER\tToolID\tEQP_PM_001\t-\t0",
    "HEADER\tLotID\tLOT_PM_001\t-\t0",
    "HEADER\tCreatedBy\tCREATOR_PM_001\t-\t0",
    "CONDITION\tDownForce\t3.5\tpsi\t0",
    "CONDITION\tPlatenSpeed\t93.0\trpm\t0",
    "CONDITION\tCarrierSpeed\t87.0\trpm\t0",
    "CONDITION\tSlurryFlow\t200.0\tml/min\t0",
    "STEP1\tPreclean\tDIWater\tlpm\t30",
    "STEP2\tCMP_Main\tSlurryA\tml/min\t120",
    "STEP3\tCMP_Over\tSlurryA\tml/min\t30",
    "STEP4\tPostclean\tDIWater\tlpm\t60",
    "TARGET\tRemovalRate\t200.0\tnm/min\t0",
    "TARGET\tUniformity\t3.0\tpct\t0",
    "TARGET\tSelectivity\t50.0\t-\t0",
]))

# 7 — JSON (anneal recipe with novel tokens for LLM)
write("rd_07_annealtech_recipe_details.json", json.dumps({
    "RecipeID": "RCP_AT_RTP_001",
    "ToolID": "EQP_AT_001",
    "LotID": "LOT_AT_001",
    "ProcessType": "Rapid Thermal Anneal",
    "CreatedBy": "CREATOR_AT_001",
    "Timestamp": "2026-02-18T11:00:00Z",
    "SpikeCycle": {
        "RampRate":   100.0,
        "SpikeTemp":  1050.0,
        "SoakTime":   0,
        "CoolRate":   80.0,
    },
    "SoakCycle": {
        "RampRate":   50.0,
        "SoakTemp":   850.0,
        "SoakTime":   30,
        "Atmosphere": "N2",
        "N2Flow":     10.0,
    },
    "PyrometerReadings": [
        {"Time": 0.0,  "Temp": 25.0,   "SensorID": "SENSOR_0001"},
        {"Time": 5.0,  "Temp": 525.0,  "SensorID": "SENSOR_0001"},
        {"Time": 10.0, "Temp": 1050.0, "SensorID": "SENSOR_0001"},
        {"Time": 10.1, "Temp": 1049.5, "SensorID": "SENSOR_0001"},
        {"Time": 15.0, "Temp": 650.0,  "SensorID": "SENSOR_0001"},
    ],
    "ExpirationDate": "2027-02-18",
    "QualityTarget": {"SheetResistance": 55.0, "JunctionDepth": 30.0}
}, indent=2))

# 8 — INI (thermal oxidation recipe)
write("rd_08_thermaldyn_recipe_details.ini", """
[RecipeHeader]
RecipeID = RCP_TD_OX_001
ToolID = EQP_TD_001
LotID = LOT_TD_001
ProcessType = Dry Thermal Oxidation
CreatedBy = CREATOR_TD_001
ExpirationDate = 2027-05-31

[Step_LOAD]
Temperature = 800.0
N2Flow = 10.0
Duration = 300
Pressure = 760.0

[Step_RAMP]
StartTemp = 800.0
EndTemp = 1000.0
RampRate = 10.0
N2Flow = 10.0
Duration = 120

[Step_OXIDIZE]
Temperature = 1000.0
O2Flow = 5.0
N2Flow = 2.0
Duration = 3600
TargetThickness = 20.0

[Step_ANNEAL]
Temperature = 1000.0
N2Flow = 10.0
Duration = 600

[Step_COOLDOWN]
StartTemp = 1000.0
EndTemp = 800.0
RampRate = 5.0
N2Flow = 10.0
Duration = 240

[QualityTargets]
ThicknessNm = 20.0
ThicknessTolerance = 0.5
RefractiveIndex = 1.462
""")

# 9 — XML (metrology recipe — novel tokens to trigger LLM)
write("rd_09_metrovision_recipe_details.xml", """<?xml version="1.0" encoding="UTF-8"?>
<MetrologyRecipe RecipeID="RCP_MV_OCD_001" ToolID="EQP_MV_001">
  <Header>
    <LotID>LOT_MV_001</LotID>
    <WaferID>WFR_MV_001</WaferID>
    <CreatedBy>CREATOR_MV_001</CreatedBy>
    <MachineType>MTYPE_OCD_001</MachineType>
    <CustomerID>CUST_MV_001</CustomerID>
    <ExpirationDate>2027-06-30</ExpirationDate>
  </Header>
  <MeasurementSites>
    <Site id="1" x="-120.0" y="0.0" label="LEFT"/>
    <Site id="2" x="0.0"    y="0.0" label="CENTER"/>
    <Site id="3" x="120.0"  y="0.0" label="RIGHT"/>
    <Site id="4" x="0.0"    y="-120.0" label="BOTTOM"/>
    <Site id="5" x="0.0"    y="120.0"  label="TOP"/>
  </MeasurementSites>
  <TargetParameters>
    <Parameter name="CDtarget"     value="45.0"   unit="nm"  tolerance="2.0"/>
    <Parameter name="SWAtarget"    value="88.0"   unit="deg" tolerance="1.5"/>
    <Parameter name="HtargetFilm" value="100.0"  unit="nm"  tolerance="3.0"/>
    <Parameter name="Roughness"    value="0.3"    unit="nm"  tolerance="0.1"/>
  </TargetParameters>
  <LegendreCoefficients>
    <SetPoint>1.50949e-07</SetPoint>
    <SetPoint>-2.07e-04</SetPoint>
    <SetPoint>5.00e-08</SetPoint>
  </LegendreCoefficients>
</MetrologyRecipe>
""")

# 10 — YAML (CMP with novel process tokens to trigger LLM adaptive loop)
write("rd_10_cleantech_recipe_details.yaml", """
RecipeID: RCP_CT_CLEAN_001
ToolID: EQP_CT_001
LotID: LOT_CT_001
ProcessType: Wafer Clean
CreatedBy: CREATOR_CT_001
ExpirationDate: 2027-07-31

CleanSequence:
  - StepID: SC1_CLEAN
    Chemical: NH4OH_H2O2_H2O
    Ratio: 1_2_10
    Temperature: 70.0
    Duration: 600
    MegasonicPower: 500.0

  - StepID: HF_DIP
    Chemical: HF_H2O
    Ratio: 1_100
    Temperature: 22.0
    Duration: 60
    MegasonicPower: 0.0

  - StepID: SC2_CLEAN
    Chemical: HCl_H2O2_H2O
    Ratio: 1_2_10
    Temperature: 70.0
    Duration: 600
    MegasonicPower: 300.0

  - StepID: FINAL_RINSE
    Chemical: DIWater
    Ratio: pure
    Temperature: 22.0
    Duration: 300
    DIWaterResistivity: 18.2

  - StepID: SPIN_DRY
    SpinSpeed: 2000.0
    N2Flow: 20.0
    Temperature: 25.0
    Duration: 120

QualityTargets:
  ParticleAdder: 5
  MetalContam: 1.00e10
  WatermarkFree: true
""")

print(f"\nDone. Created 20 test files in {OUT}/")
print("\nMachine constant files:")
for f in sorted(OUT.glob("mc_*.* ")):
    print(f"  {f.name}")
print("\nRecipe detail files:")
for f in sorted(OUT.glob("rd_*.*")):
    print(f"  {f.name}")