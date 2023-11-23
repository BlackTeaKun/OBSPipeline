#include <tbb/flow_graph.h>
#include <iostream>

#include <chrono>
#include <thread>

int rev(int x){
    std::cout << x << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return -x;
}

class Driver{
    public:
    Driver(int _x):x(_x){
        tbb::flow::function_node<int, int> f(g, 10, [this](int y){return this->add(y);});
    }
    ~Driver(){
        g.wait_for_all();
    }
    private:
    int add(int y){
        return y + x;
    }
    int x;
    tbb::flow::graph g;
};

int main(){
    tbb::flow::graph g;
    tbb::flow::function_node<int, int> f(g, 10, rev);
    tbb::flow::function_node<int> w(g, 1, [](int x){
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::cout << "recived " << x << std::endl;
        });
    tbb::flow::make_edge(f, w);
    for(int i = 1; i < 100; ++i){
        f.try_put(-i);
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    g.wait_for_all();
    f.try_put(-300);
    g.wait_for_all();
    g.wait_for_all();

    return 0;
}