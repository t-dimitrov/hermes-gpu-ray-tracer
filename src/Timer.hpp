#pragma once
#include <chrono>
#include <string>
#include <iostream>

namespace Hermes
{
    template <typename Resolution = std::chrono::milliseconds>
    class Timer
    {
    public:
        using Clock = std::chrono::high_resolution_clock;

        Timer(const std::string& debugName = "Default Timer")
            : _startTimepoint(Clock::now())
            , _debugName(debugName)
            , _didStop(false)
        {}

        ~Timer()
        {
            if (!_didStop)
            {
                Stop();
            }
        }

        void Stop()
        {
            Clock::time_point endTimepoint = Clock::now();
            auto duration = std::chrono::duration_cast<Resolution>(endTimepoint - _startTimepoint);
            std::cout << _debugName << ": " << duration.count() << "ms (" << duration.count() / 1000 << "s)" << std::endl;

            _didStop = true;
        }

    private:
        Clock::time_point _startTimepoint;
        std::string _debugName;
        bool _didStop;
    };
}
