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

#include <cstdint>
#include <immintrin.h>
#include <string>
#include <iostream>
namespace tiny_dnn {
struct half;
}
namespace std {
const tiny_dnn::half abs(const tiny_dnn::half &h);
const tiny_dnn::half exp(const tiny_dnn::half &h);
const tiny_dnn::half sqrt(const tiny_dnn::half &h);
const tiny_dnn::half log(const tiny_dnn::half &h);
//TODO(Randl): uniform_rand
}

namespace tiny_dnn {

constexpr uint_least16_t CNN_HALF_SIGN = 1 << 15;
constexpr uint_least16_t CNN_HALF_EXPONENT = 0x1F << 10;
constexpr uint_least16_t CNN_HALF_MANTISSA = 0x3FF;

struct half {

  // Constructors
  constexpr half() : _h() {};  // no initialization
  half(float f) : _h(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), 0))) {};
  //half(double d) : _h(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(float(d)), 0))) {};
  //half(long double l) : _h(half(double(l))._h) {};
  explicit half(uint_least16_t u, bool x) : _h(u) {};
  /*half(uint_fast64_t u) : _h(half(double(u))._h) {};
  half(int_fast64_t i) : _h(half(double(i))._h) {};*/

  // checks
  bool isFinite() const;
  bool isNormalized() const;
  bool isDenormalized() const;
  bool isZero() const;
  bool isNan() const;
  bool isInfinity() const;
  bool isNegative() const;

  // operators
  half operator+() const;  // unary +
  half operator-() const;  // unary -
  bool operator!() const;  // logical not
  void operator+=(const half &i);
  void operator-=(const half &i);
  void operator/=(const half &i);
  void operator*=(const half &i);
  friend const half operator+(const half &left, const half &right);
  friend const half operator-(const half &left, const half &right);
  friend const half operator/(const half &left, const half &right);
  //friend const half operator/(const half &left, const uint_fast64_t &right);
  //friend const half operator/(const half &left, const int_fast64_t &right);
  friend const half operator*(const half &left, const half &right);
  friend const bool operator==(const half &left, const half &right);
  friend const bool operator!=(const half &left, const half &right);
  friend const bool operator>(const half &left, const half &right);
  friend const bool operator<(const half &left, const half &right);
  friend const bool operator>=(const half &left, const half &right);
  friend const bool operator<=(const half &left, const half &right);
  friend const bool operator&&(const half &left, const half &right);
  friend const bool operator||(const half &left, const half &right);
  friend std::ostream &operator<<(std::ostream &os, const half &h);
  friend std::istream &operator>>(std::istream &is, half &h);

  operator double() const;
 /* operator float() const;
  operator int() const;
  operator bool() const;
  operator uint8_t() const;*/

  // operations
  friend const half std::abs(const half &h);

 private:
  uint_least16_t _h;
};

half half::operator+() const {
  return *this;
}
half half::operator-() const {
  return half(uint_least16_t(_h ^ CNN_HALF_SIGN), true);
}
bool half::operator!() const {
  return (_h & (~CNN_HALF_SIGN)) == 0;
}
void half::operator+=(const half &i) {
  *this = *this + i;
  return;
}
void half::operator-=(const half &i) {
  *this = *this - i;
  return;
}
void half::operator/=(const half &i) {
  *this = *this / i;
  return;
}
void half::operator*=(const half &i) {
  *this = *this * i;
  return;
}
const half operator+(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return half(z[0] + z[1]);
}
const half operator-(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return half(z[0] - z[1]);
}
const half operator/(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return half(z[0] / z[1]);
}
/*const half operator/(const half &left, const uint_fast64_t &right) {
  return half(float(left) / right);
}
const half operator/(const half &left, const int_fast64_t &right) {
  return half(float(left) / right);
}*/
const half operator*(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return half(z[0] * z[1]);
}
const bool operator==(const half &left, const half &right) {
  return left._h == right._h;
}
const bool operator!=(const half &left, const half &right) {
  return left._h != right._h;
}
const bool operator>(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return z[0] > z[1];
}
const bool operator<(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return z[0] < z[1];
}
const bool operator>=(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return z[0] >= z[1];
}
const bool operator<=(const half &left, const half &right) {
  auto z = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, 0, 0, right._h, left._h));
  return z[0] <= z[1];
}
const bool operator&&(const half &left, const half &right) {
  return static_cast<bool>(left) && static_cast<bool>(right);
}
const bool operator||(const half &left, const half &right) {
  return static_cast<bool>(left) || static_cast<bool>(right);
}
std::ostream &operator<<(std::ostream &os, const half &h) {
  os << float(h);
  return os;
}
std::istream &operator>>(std::istream &is, half &h) {
  float x;
  is >> x;
  h = half(x);
  return is;
}

half::operator double() const {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(_h)));
}/*
half::operator float() const {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(_h)));
}
half::operator int() const {
  return int(float(*this));
}
half::operator bool() const {
  return (_h & (~CNN_HALF_SIGN)) != 0;
}
half::operator uint8_t() const {
  return uint8_t(float(*this));
}*/

bool half::isFinite() const {
  return ((_h & CNN_HALF_EXPONENT) ^ CNN_HALF_EXPONENT) != 0;
}
bool half::isNormalized() const {
  return this->isFinite() && !this->isDenormalized();
}
bool half::isDenormalized() const {
  return this->isFinite() && (_h & CNN_HALF_EXPONENT) == 0;
}
bool half::isZero() const {
  return !bool(*this);
}
bool half::isNan() const {
  return !this->isFinite() && (_h & CNN_HALF_MANTISSA) != 0;
}
bool half::isInfinity() const {
  return !this->isFinite() && (_h & CNN_HALF_MANTISSA) == 0;
}
bool half::isNegative() const {
  return (_h & CNN_HALF_SIGN) != 0;
}

// literal
half operator ""_h(long double d) {
  return half(double(d));
}

}

namespace std {
const tiny_dnn::half abs(const tiny_dnn::half &h) {
  return tiny_dnn::half(uint_least16_t(h._h & (~tiny_dnn::CNN_HALF_SIGN)), true);
}

const tiny_dnn::half exp(const tiny_dnn::half &h) {
  return tiny_dnn::half(exp(static_cast<float>(h)));
}
const tiny_dnn::half sqrt(const tiny_dnn::half &h) {
  return tiny_dnn::half(sqrt(static_cast<float>(h)));
}

const tiny_dnn::half log(const tiny_dnn::half &h){
  return tiny_dnn::half(log(static_cast<float>(h)));
}

template<>
struct is_floating_point<tiny_dnn::half> : std::integral_constant<bool, true> {};
template<>
struct is_arithmetic<tiny_dnn::half> : std::integral_constant<bool, true> {};
template<>
struct is_signed<tiny_dnn::half> : std::integral_constant<bool, true> {};
}
