"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class AmbientStage(RenderStage):
    """ This stage computes the ambient term """

    required_inputs = [
        "DefaultEnvmap", "PrefilteredBRDF", "PrefilteredMetalBRDF",
        "PrefilteredCoatBRDF"
    ]
    required_pipes = ["ShadedScene", "GBuffer"]

    @property
    def produced_pipes(self):
        return {
            "ShadedScene": self._target.color_tex,
            "PostAmbientScene": self._target.color_tex
        }

    def create(self):
        self._target = self.create_target("AmbientStage")
        self._target.add_color_attachment(bits=16)
        self._target.prepare_buffer()

    def reload_shaders(self):
        self._target.shader = self.load_shader("ambient_stage.frag.glsl")
