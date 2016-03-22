(defproject flowgraph "0.1.0-SNAPSHOT"
  :description "Machine learning software library written in Clojure for numerical computation using data flow graphs."
  :url "http://example.com/FIXME"
  :license {:name "The MIT License (MIT)"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [net.mikera/core.matrix "0.50.0"]
                 [net.mikera/vectorz-clj "0.43.1"]
                 [org.clojure/core.async "0.2.374"] ]
  :profiles {:dev {:plugins [[com.jakemccrary/lein-test-refresh "0.14.0"]]}})
