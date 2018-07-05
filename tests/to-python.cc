// Copyright (C) 2015 by Benjamin Chrétien, CNRS-LIRMM.
//
// This file is part of roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#include "shared-tests/fixture.hh"

#include <iostream>

#include <roboptim/core/python/to-python.hh>

using namespace roboptim;
using namespace roboptim::python;

typedef boost::shared_ptr<boost::test_tools::output_test_stream>
output_ptr;

BOOST_AUTO_TEST_SUITE (to_python)

BOOST_AUTO_TEST_CASE (redir)
{
  output_ptr output = retrievePattern ("to-python");

  ToPython tp;

  tp << "import numpy as np"
     << "np.set_printoptions(formatter={'all': lambda x: str(x)})"
     << "ar = np.zeros((3,3))"
     << "print(ar)"
     << "ar[0,1] = 42."
     << "print(ar.max())";

  // Send a wrong command
  BOOST_CHECK_THROW (tp << "foo 42", std::runtime_error);

  // Flush to output
  tp >> (*output);

  // Should do nothing
  tp >> (*output);

  std::cout << output->str () << std::endl;
  BOOST_CHECK (output->match_pattern ());
}

BOOST_AUTO_TEST_CASE (simultaneous)
{
  output_ptr output = retrievePattern ("to-python-simultaneous");

  ToPython tp1;

  tp1 << "print(\"foo\")";

  {
    ToPython tp2;
    tp2 << "print(\"bar\")";
    tp2 >> (*output);
  }

  tp1 << "print(42)";

  // Flush to output
  tp1 >> (*output);

  std::cout << output->str () << std::endl;
  BOOST_CHECK (output->match_pattern ());
}

BOOST_AUTO_TEST_SUITE_END ()
