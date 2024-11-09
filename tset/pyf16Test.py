import pyf16

aero_model = pyf16.AerodynamicModel("./models/f16_model")
aero_model.install("./models/f16_model/data")
control_limits = aero_model.load_ctrl_limits()
print(f"Control Limits: {control_limits.thrust_cmd_limit_top}")

trim_target = pyf16.TrimTarget(15000, 500, None, None)
trim_init = None
trim_result = pyf16.trim(aero_model, trim_target, control_limits, trim_init)

f16 = pyf16.PlaneBlock("1", aero_model, trim_result, [0, 0, 0], control_limits)
print(f"f16 State: {f16.state.state.to_list()}")
for i in range(10000):
    core_output = f16.update(
        pyf16.Control(thrust=100, elevator=0, aileron=0, rudder=0), i*0.01
    )
    print(core_output.state.to_list()[0:3])
# core_output = f16.update(
#     pyf16.Control(thrust=100, elevator=0, aileron=0, rudder=0), 0.1
# )
# print(core_output.state.to_list())

f16.delete_model()
aero_model.uninstall()
