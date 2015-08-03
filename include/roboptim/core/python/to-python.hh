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

#ifndef ROBOPTIM_CORE_PYTHON_TO_PYTHON_HH
# define ROBOPTIM_CORE_PYTHON_TO_PYTHON_HH

# include <string>

namespace roboptim
{
  namespace python
  {
    /// \brief Helper to run Python commands in the Python interpreter.
    /// WARNING: this is not thread-safe for now (GIL etc.).
    class ToPython
    {
    public:

      /// \brief Constructor.
      /// Initializes the Python interpreter.
      ToPython ();

      /// \brief Destructor.
      /// This exits the Python interpreter.
      ~ToPython ();

      /// \brief Run a command in the Python interpreter.
      /// \param cmd command to run.
      const ToPython& operator << (const char* cmd) const;

      /// \brief Run a command in the Python interpreter.
      /// \param cmd command to run.
      const ToPython& operator << (const std::string& cmd) const;

      /// \brief Flush the output of the Python interpreter.
      /// \param o output stream.
      void operator >> (std::ostream& o);

    private:
      /// \brief Number of instances.
      static int instances_;

      /// \brief Buffering callback.
      /// \param s string added to the buffer.
      void buffering (const std::string& s);

      /// \brief Buffer.
      std::string buffer_;
    };
  } // end of namespace python
} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PYTHON_TO_PYTHON_HH
