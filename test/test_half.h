/*
    Copyright (c) 2016, Evgenii Zheltonozhskii
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/util/half.h"

namespace tiny_dnn {
TEST(arithm, half_float) {
  half abs_error(1e-2);
  EXPECT_EQ(half(7.2), std::abs(half(-7.2)));
  EXPECT_EQ(half(7.2), std::abs(half(7.2)));
  EXPECT_EQ(false, bool(half(0.0)));
  EXPECT_NEAR(half(8.602), half(7.238) + half(1.364), abs_error);
  EXPECT_NEAR(half(8.602), half(7.238) + half(1.364), abs_error);
  EXPECT_NEAR(half(5.874), half(7.238) - half(1.364), abs_error);
  EXPECT_NEAR(half(9.872632), half(7.238) * half(1.364), abs_error);
  EXPECT_NEAR(half(5.30645), half(7.238) / half(1.364), abs_error);
  EXPECT_EQ(5, int(half(5.0)));
  EXPECT_EQ(false, bool(half(0.0)));
  half x = 5.0;
  EXPECT_NEAR(half(5.0), x, abs_error);
  x += 1000.0;
  EXPECT_NEAR(double(1005.0), double(x), 1e-2);
  x += 5000.0;
  EXPECT_NEAR(double(6005.0), double(x), 1);
  x /= half(3000.0);
  EXPECT_NEAR(double(2.002), double(x), 1e-2);
  half y(std::numeric_limits<float>::infinity()), z = half(1.0), w = half(0.0);
  EXPECT_EQ(true, y.isInfinity());
  EXPECT_EQ(true, x.isFinite());
  EXPECT_EQ(true, w.isZero());
  EXPECT_EQ(false, z.isNan());
  z = w / w;
  EXPECT_EQ(true, z.isNan());
  EXPECT_EQ(true, (-x).isNegative());
  EXPECT_EQ(true, half(5.0).isNormalized());
  EXPECT_EQ(true, half(0.00002).isDenormalized());
  EXPECT_EQ(0.654_h, half(0.654));
}
}
