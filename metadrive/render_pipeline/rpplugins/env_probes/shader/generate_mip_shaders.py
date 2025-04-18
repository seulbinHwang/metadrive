import os
from os.path import dirname, realpath, isdir

os.chdir(dirname(realpath(__file__)))

if not isdir("mips"):
    os.makedirs("mips")

for i in range(12):
    num_samples = 256 + i * 12
    if i == 0:
        num_samples = 128
    with open('mips/{}.autogen.glsl'.format(i), 'w') as handle:
        handle.write('// Autogenerated, do not edit\n')
        handle.write('#version 430\n')
        handle.write('#define SHADER_NUM_SAMPLES {}\n'.format(num_samples))
        handle.write('#pragma include "../filter_cubemap.frag.glsl"\n')
