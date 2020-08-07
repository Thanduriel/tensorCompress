#pragma once

#include <ostream>

template<typename T, size_t Size>
std::ostream& operator<<(std::ostream& os, std::array<T,Size> const& arr) 
{
	os << "[";
	for (auto& el : arr)
		os << el << ",";
	return os << "]";
}