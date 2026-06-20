// swift-tools-version:6.2
//
// NOTE: the spec text says "swift-tools-version:6.0", but `.macOS(.v26)` (a
// non-negotiable requirement for the Liquid Glass deployment target) is only
// available in PackageDescription 6.2+ — under 6.0 the manifest fails with
// "'v26' is unavailable". 6.2 is the minimum tools version that exposes
// `.macOS(.v26)`; the Swift 6 LANGUAGE MODE is still pinned below via
// `.swiftLanguageMode(.v6)`, which is what the spec's intent ("Swift 6 strict
// concurrency") actually requires.
import PackageDescription

let package = Package(
    name: "Fever",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .executable(name: "Fever", targets: ["Fever"])
    ],
    targets: [
        // All pure / non-UI logic lives in a library so BOTH the app executable
        // and the headless test runner can link it. The app's @main + SwiftUI
        // views stay in the `Fever` executable target below (unchanged
        // sources, they just `import FeverCore`).
        //
        // WHY a library split (mechanism B): on this Command-Line-Tools-only
        // toolchain `swift test` cannot run — there is no `xctest` host tool and
        // the .xctest bundle fails codesign ("resource fork / Finder info") in
        // this file-provider-synced directory. So tests run as a normal
        // executable (`swift run FeverCheck`) that links the same library,
        // asserts, and exits non-zero on failure.
        .target(
            name: "FeverCore",
            path: "Fever/Sources",
            exclude: ["App"],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .executableTarget(
            name: "Fever",
            dependencies: ["FeverCore"],
            path: "Fever/Sources/App",
            // -parse-as-library: @main lives in FeverMain.swift, not main.swift.
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
        .executableTarget(
            name: "FeverCheck",
            dependencies: ["FeverCore"],
            path: "Tests/Check",
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
