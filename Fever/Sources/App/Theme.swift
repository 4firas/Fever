import FeverCore
import SwiftUI

/// Central kurokula palette + reusable view helpers so the whole Fever UI
/// shares one cohesive, dark/warm/professional look. Crimson is the signature
/// accent; numeric telemetry is always monospaced-digit.
///
/// This is a pure styling file: it never touches the tracking pipeline, config
/// values, or any binding semantics.
enum Theme {

    // MARK: Palette (kurokula)

    /// Charcoal app background (#131515).
    static let background       = Color(hex: 0x131515)
    /// Elevated surface for cards / panels (#181a1a).
    static let surface          = Color(hex: 0x181a1a)
    /// A slightly lifted surface for nested rows.
    static let surfaceRaised    = Color(hex: 0x1f2222)

    /// Primary warm-beige text (#dfcfc2).
    static let textPrimary      = Color(hex: 0xdfcfc2)
    /// Secondary / muted text (#9a8e84).
    static let textSecondary    = Color(hex: 0x9a8e84)
    /// Tertiary / faint text (#505151).
    static let textMuted        = Color(hex: 0x505151)

    /// Primary signature accent crimson (#791c1c).
    static let crimson          = Color(hex: 0x791c1c)
    /// Brighter interactive / hover crimson (#c35951).
    static let crimsonBright     = Color(hex: 0xc35951)
    /// Secondary dusty-rose (#ac756f).
    static let dustyRose        = Color(hex: 0xac756f)

    /// Tracking / good neon green (#aeffa4).
    static let good             = Color(hex: 0xaeffa4)
    /// Lost / error soft red (#c35951).
    static let lost             = Color(hex: 0xc35951)
    /// Acquiring / warning yellow (#fff600) — use sparingly.
    static let warning          = Color(hex: 0xfff600)
    /// Info cyan (#8fcfcf).
    static let info             = Color(hex: 0x8fcfcf)
    /// Calm blue (#5f7faf).
    static let calmBlue         = Color(hex: 0x5f7faf)

    /// PinoQuest-style skeleton joint coloring: cyan = body-LEFT, orange = body-RIGHT.
    static let trackerLeft      = Color(hex: 0x4fd6e6)
    static let trackerRight     = Color(hex: 0xff9d42)

    // MARK: Type scale (12–16pt, clear weight hierarchy)

    /// Card / panel titles.
    static let titleFont        = Font.system(size: 15, weight: .semibold)
    /// Muted section headers (sidebar / inspector group captions).
    static let sectionFont      = Font.system(size: 11, weight: .semibold).width(.standard)
    /// Row labels.
    static let labelFont        = Font.system(size: 13, weight: .regular)
    /// Numeric / telemetry / address values — always monospaced digits.
    static let valueFont        = Font.system(size: 13, weight: .medium).monospacedDigit()
    /// Small monospaced addresses / hosts.
    static let monoSmall        = Font.system(size: 11, weight: .regular, design: .monospaced)
    /// Tiny captions.
    static let captionFont      = Font.system(size: 11, weight: .regular)
}

// MARK: - Hex color init

extension Color {
    /// Build an opaque Color from a 0xRRGGBB integer.
    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xff) / 255
        let g = Double((hex >> 8) & 0xff) / 255
        let b = Double(hex & 0xff) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }
}

// MARK: - Tracker state

/// A tracker's live state, used for status dots / pill tints across the UI.
enum TrackerState {
    case tracking   // strong live confidence
    case acquiring  // weak live confidence
    case lost       // enabled but no live data
    case idle       // not enabled

    /// Maps a live joint (or its absence) + enabled flag into a state.
    static func resolve(live: VRJoint?, enabled: Bool) -> TrackerState {
        guard enabled else { return .idle }
        guard let live else { return .lost }
        switch live.confidence {
        case ..<0.4:  return .lost
        case ..<0.75: return .acquiring
        default:      return .tracking
        }
    }

    var color: Color {
        switch self {
        case .tracking:  return Theme.good
        case .acquiring: return Theme.warning
        case .lost:      return Theme.lost
        case .idle:      return Theme.textMuted
        }
    }
}

// MARK: - Reusable view helpers

/// A muted, uppercased section header in beige — used for sidebar / inspector
/// group captions to give consistent visual hierarchy.
struct SectionHeader: View {
    let title: String
    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title.uppercased())
            .font(Theme.sectionFont)
            .tracking(0.8)
            .foregroundStyle(Theme.textSecondary)
    }
}

/// A small filled status dot tinted by tracker state, with a soft glow ring.
struct StatusDot: View {
    let color: Color
    var size: CGFloat = 8

    init(_ color: Color, size: CGFloat = 8) {
        self.color = color
        self.size = size
    }
    init(state: TrackerState, size: CGFloat = 8) {
        self.color = state.color
        self.size = size
    }

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: size, height: size)
            .overlay(
                Circle().stroke(color.opacity(0.35), lineWidth: size * 0.45)
            )
    }
}

/// A label/value telemetry row with aligned, monospaced value text.
struct MetricRow: View {
    let label: String
    let value: String
    var valueColor: Color = Theme.textPrimary
    var mono: Bool = true

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(Theme.labelFont)
                .foregroundStyle(Theme.textSecondary)
            Spacer(minLength: 12)
            Text(value)
                .font(mono ? Theme.valueFont : Theme.labelFont)
                .foregroundStyle(valueColor)
                // Reflow gracefully on a narrow sidebar instead of clipping the
                // value (e.g. the OSC `host:port`): keep it on one line but let
                // it scale down before it would truncate.
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .truncationMode(.middle)
        }
    }
}

/// A confidence bar tinted green→red, used in the inspector tracker cards.
struct ConfidenceBar: View {
    let confidence: Float   // 0...1

    private var tint: Color {
        switch confidence {
        case ..<0.4:  return Theme.lost
        case ..<0.75: return Theme.warning
        default:      return Theme.good
        }
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Theme.textMuted.opacity(0.25))
                Capsule()
                    .fill(tint)
                    .frame(width: max(0, min(1, CGFloat(confidence))) * geo.size.width)
            }
        }
        .frame(height: 5)
    }
}